"""Export cached IAAI flow diagnostics on CPU; optional aligned GT gives EPE."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from blur_gs.cache import FlowCache


def flow_rgb(flow, scale):
    """HSV: hue=direction in image coordinates (+x right, +y down), value=speed."""
    magnitude = np.linalg.norm(flow, axis=0)
    hue = np.mod(np.arctan2(flow[1], flow[0]), 2 * np.pi) / (2 * np.pi)
    hsv = np.stack([hue * 255, np.full_like(hue, 255),
                    np.clip(magnitude / scale, 0, 1) * 255], axis=-1).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def heatmap(values, scale, valid):
    v = np.clip(values / scale, 0, 1)
    rgb = np.stack([v, v ** 2, np.zeros_like(v)], axis=-1)
    rgb[~valid] = .15
    return Image.fromarray((rgb * 255).astype(np.uint8))


def endpoint_error(prediction, target, pred_mask, target_mask):
    if prediction.shape != target.shape or pred_mask.shape != target_mask.shape:
        raise ValueError("GT must have the same flow grid as prediction; no implicit resizing")
    valid = (pred_mask > 0) & (target_mask > 0)
    error = np.linalg.norm(prediction - target, axis=0)
    return error, valid, float(error[valid].mean()) if valid.any() else None


def source_image(entry, images_root):
    source = entry.get("source")
    if not source:
        raise ValueError("Cache entry lacks source image path")
    original_path = Path(source)
    if images_root is not None:
        # Explicit override is useful after copying a cache to a different host.
        relative = Path(original_path.name) if original_path.is_absolute() else original_path
        path = (images_root / relative).resolve()
        if not path.is_relative_to(images_root.resolve()):
            raise ValueError("Source path escapes --images")
    else:
        if not original_path.is_absolute():
            raise ValueError("Relative source paths require --images /original/image/directory")
        path = original_path
    with Image.open(path) as img:
        result = img.convert("RGB")
    crop = entry["crop"]
    if result.size != (crop["original_width"], crop["original_height"]):
        raise ValueError(f"Source size differs from cache: {path}")
    return result


def arrows(image, flow, mask, spacing, factor):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    h, w = mask.shape
    for y in range(spacing // 2, h, spacing):
        for x in range(spacing // 2, w, spacing):
            if mask[y, x] <= 0:
                continue
            vector = flow[:, y, x] * factor
            length = float(np.linalg.norm(vector))
            if length < .25:
                continue
            end = np.array([x, y]) + vector
            unit = vector / length
            side = np.array([-unit[1], unit[0]])
            tip = min(4., length * .4)
            draw.line([(x, y), tuple(end)], fill=(255, 220, 30), width=1)
            draw.polygon([tuple(end), tuple(end - tip * unit + tip * .5 * side),
                          tuple(end - tip * unit - tip * .5 * side)], fill=(255, 220, 30))
    return image


def save_sheet(panels, path, name, footer):
    width, height = 400, 320
    rows = (len(panels) + 2) // 3
    sheet = Image.new("RGB", (3 * width, rows * height + 72), (20, 24, 32))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), name[:150], fill="white")
    for index, (label, picture) in enumerate(panels):
        x, y = (index % 3) * width, 32 + (index // 3) * height
        draw.text((x + 10, y + 4), label, fill="white")
        picture = picture.copy()
        picture.thumbnail((width - 20, height - 40))
        sheet.paste(picture, (x + (width - picture.width) // 2, y + 28))
    draw.text((12, rows * height + 38), footer, fill=(255, 210, 100))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cache_source = parser.add_mutually_exclusive_group(required=True)
    cache_source.add_argument("--flow_cache", type=Path, help="Directory containing manifest.json")
    cache_source.add_argument("--model_path", "-m", type=Path, help="Read cache location from blur_gs_config.json")
    parser.add_argument("--images", type=Path, help="Override original image directory after moving data/cache")
    parser.add_argument("--output", type=Path, default=Path("output/flow_diagnostics"))
    parser.add_argument("--limit", type=int, default=12, help="Evenly spaced sorted images; 0=all")
    parser.add_argument("--max_flow", type=float, default=20., help="Shared color scale in network pixels")
    parser.add_argument("--max_error", type=float, default=5., help="GT error heatmap maximum in network pixels")
    parser.add_argument("--arrow_scale", type=float, default=2., help="Display-only arrow length multiplier")
    parser.add_argument("--spacing", type=int, default=20)
    parser.add_argument("--gt_cache", type=Path, help="Optional actual GT in the same cache format/grid/crop/direction")
    args = parser.parse_args()
    if (args.limit < 0 or args.spacing < 1 or
            any(not np.isfinite(v) or v <= 0 for v in (args.max_flow, args.max_error, args.arrow_scale))):
        parser.error("Scales/spacing must be positive finite values and limit nonnegative")
    if args.model_path:
        config = json.loads((args.model_path / "blur_gs_config.json").read_text(encoding="utf-8"))
        args.flow_cache = Path(config["flow_cache"])
    cache = FlowCache(args.flow_cache)
    names = sorted(cache.entries)
    if not names:
        parser.error("Cache contains no images")
    if args.limit and len(names) > args.limit:
        names = [names[i] for i in np.linspace(0, len(names) - 1, args.limit, dtype=int)]
    gt_cache = FlowCache(args.gt_cache, names) if args.gt_cache else None
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, name in enumerate(names):
        flow, mask, crop = cache.get(name, "cpu")
        flow, mask = flow.numpy(), mask.numpy()
        valid = mask > 0
        magnitude = np.linalg.norm(flow, axis=0)
        original = source_image(cache.entries[name], args.images)
        box = (crop["left"], crop["top"], crop["left"] + crop["width"], crop["top"] + crop["height"])
        image = original.crop(box).resize((flow.shape[2], flow.shape[1]), Image.Resampling.BILINEAR)
        annotated = original.copy()
        ImageDraw.Draw(annotated).rectangle(box, outline="yellow", width=max(1, original.width // 200))
        panels = [("Input image / yellow = estimator crop", annotated),
                  ("Cropped input (network grid)", image),
                  (f"Flow arrows x{args.arrow_scale:g} (display only)", arrows(image, flow, mask, args.spacing, args.arrow_scale)),
                  (f"Direction color; brightness 0..{args.max_flow:g} px", flow_rgb(flow, args.max_flow)),
                  (f"Magnitude: black=0, yellow>={args.max_flow:g} px", heatmap(magnitude, args.max_flow, valid)),
                  ("Cache mask: validity, NOT confidence", Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB"))]
        record = dict(name=name, cache_mask_fraction=float(valid.mean()),
                      mean_magnitude_px=float(magnitude[valid].mean()) if valid.any() else None,
                      p95_magnitude_px=float(np.percentile(magnitude[valid], 95)) if valid.any() else None,
                      gt_valid_pixels=None, gt_epe_px=None)
        if gt_cache:
            gt, gt_mask, gt_crop = gt_cache.get(name, "cpu")
            if crop != gt_crop:
                raise ValueError(f"GT crop mismatch for {name}; align coordinates before evaluation")
            gt, gt_mask = gt.numpy(), gt_mask.numpy()
            error, common, record["gt_epe_px"] = endpoint_error(flow, gt, mask, gt_mask)
            record["gt_valid_pixels"] = int(common.sum())
            panels.extend([(f"GT flow; same 0..{args.max_flow:g} px scale", flow_rgb(gt, args.max_flow)),
                           (f"EPE: black=0, yellow>={args.max_error:g} px", heatmap(error, args.max_error, common)),
                           ("GT / prediction common valid pixels", Image.fromarray((common * 255).astype(np.uint8)).convert("RGB"))])
            gt_cache.loaded.clear()
        token = hashlib.sha256(name.encode()).hexdigest()[:12]
        record["image"] = f"{index:04d}_{token}.png"
        footer = "EPE uses supplied direction; no automatic sign choice." if gt_cache else "No ground truth: these are diagnostics, NOT an accuracy score."
        save_sheet(panels, args.output / record["image"], name, footer)
        records.append(record)
        cache.loaded.clear()
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "report.json").write_text(json.dumps(dict(
        flow_cache=str(args.flow_cache.resolve()), gt_cache=str(args.gt_cache) if args.gt_cache else None,
        units="network_pixels", max_flow=args.max_flow, max_error=args.max_error,
        arrow_scale=args.arrow_scale, direction="as_provided_no_sign_alignment",
        note="Cache mask is not confidence; no GT means no accuracy estimate.", images=records), indent=2), encoding="utf-8")
    print(f"Saved {len(records)} diagnostic sheets and summaries to {args.output.resolve()}")


if __name__ == "__main__":
    main()
