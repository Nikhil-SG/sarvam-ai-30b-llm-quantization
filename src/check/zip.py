import os
import zipfile
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ── File types that compress well (text, configs, logs, etc.) ─────────────────
COMPRESSIBLE_EXTENSIONS = {
    ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".py", ".sh", ".md", ".log", ".csv", ".xml", ".html",
}

# ── Binary / already-packed model files — storing is much faster ──────────────
# ZIP_STORED skips compression entirely; these files shrink < 1% anyway.
SKIP_COMPRESS_EXTENSIONS = {
    ".bin", ".pt", ".pth", ".safetensors", ".gguf", ".ggml",
    ".npz", ".npy", ".pkl", ".pickle", ".h5", ".hdf5",
    ".onnx", ".tflite", ".pb", ".model", ".weights",
    ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".zip", ".gz",
}


def get_folder_size(folder_path, skip_extensions=None):
    """Calculate total size of a folder and all its contents."""
    skip_extensions = skip_extensions or set()
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if Path(filename).suffix.lower() in skip_extensions:
                continue
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def count_items(folder_path, skip_extensions=None):
    """Count total files and folders in a directory."""
    skip_extensions = skip_extensions or set()
    count = 0
    for _, dirnames, filenames in os.walk(folder_path):
        count += len(dirnames)
        count += sum(
            1 for f in filenames
            if Path(f).suffix.lower() not in skip_extensions
        )
    return count


def _compress_file_to_bytes(args):
    """
    Worker: compress a single file to bytes in a subprocess.
    Returns (arcname, compressed_bytes, compress_type, error) so the
    main process can write everything sequentially into the zip.
    """
    file_path, arcname, compress_type, compress_level = args
    try:
        import io
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w",
                             compression=compress_type,
                             compresslevel=compress_level) as tmp:
            tmp.write(file_path, arcname)
        mem.seek(0)
        raw = mem.read()
        return arcname, raw, compress_type, None
    except Exception as e:
        return arcname, None, compress_type, str(e)


def _choose_compression(file_path):
    """Pick compression strategy based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext in SKIP_COMPRESS_EXTENSIONS:
        return zipfile.ZIP_STORED, None     # no compression
    elif ext in COMPRESSIBLE_EXTENSIONS:
        return zipfile.ZIP_DEFLATED, 6      # balanced
    else:
        return zipfile.ZIP_STORED, None     # unknown → store (safe & fast)


def folder_to_zip(folder_path, workers=None, skip_extensions=None, zip_name=None):
    """
    Convert a folder and all its contents to a zip file.

    Speedups vs original:
      • ZIP_STORED for binary/model files  → skips useless compression
      • ProcessPoolExecutor for compressible files → uses all CPU cores

    Args:
        folder_path (str)     : Path to the folder to be zipped.
        workers (int | None)  : Parallel worker count. Defaults to os.cpu_count().
        skip_extensions (set) : Extensions to exclude entirely (e.g. {'.safetensors'}).
        zip_name (str | None) : Override output filename. Defaults to '<folder>.zip'.
    """
    folder_path     = Path(folder_path).resolve()
    workers         = workers or os.cpu_count()
    skip_extensions = {ext.lower() for ext in (skip_extensions or set())}

    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return

    # ── Output filename ───────────────────────────────────────────────────────
    zip_filename = zip_name if zip_name else f"{folder_path.name}.zip"
    zip_filepath = folder_path.parent / zip_filename

    # ── Stats ─────────────────────────────────────────────────────────────────
    print("\n📊 Calculating folder statistics...")
    if skip_extensions:
        print(f"⏭️  Skipping extensions : {', '.join(sorted(skip_extensions))}")

    folder_size_bytes = get_folder_size(folder_path, skip_extensions)
    folder_size_mb    = folder_size_bytes / (1024 * 1024)
    total_items       = count_items(folder_path, skip_extensions)

    print(f"📁 Folder size  : {folder_size_mb:.2f} MB  (after skips)")
    print(f"📄 Total items  : {total_items} (files + folders)")
    print(f"⚙️  Workers      : {workers} parallel processes")
    print(f"💾 Output file  : {zip_filepath.name}")
    print(f"\n🔄 Starting compression...\n")

    try:
        start_time = time.time()

        # ── Collect all items ─────────────────────────────────────────────────
        dirs_to_add    = []   # (dir_path, arcname)
        files_stored   = []   # written directly — ZIP_STORED (large binaries)
        files_parallel = []   # deflated in parallel — ZIP_DEFLATED (text/configs)

        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                arcname  = str(dir_path.relative_to(folder_path.parent)) + "/"
                dirs_to_add.append((dir_path, arcname))

            for file in files:
                file_path = Path(root) / file

                # ── Skip unwanted extensions entirely ─────────────────────────
                if file_path.suffix.lower() in skip_extensions:
                    continue

                arcname       = str(file_path.relative_to(folder_path.parent))
                compress_type, level = _choose_compression(file_path)

                if compress_type == zipfile.ZIP_STORED:
                    files_stored.append((file_path, arcname))
                else:
                    files_parallel.append((str(file_path), arcname,
                                           compress_type, level))

        total_steps = len(dirs_to_add) + len(files_stored) + len(files_parallel)

        with zipfile.ZipFile(zip_filepath, "w", allowZip64=True) as zipf:
            with tqdm(total=total_steps, desc="Compressing",
                      unit="item", colour="green") as pbar:

                # 1️⃣  Directory entries
                for dir_path, arcname in dirs_to_add:
                    info = zipfile.ZipInfo(arcname)
                    zipf.writestr(info, "")
                    pbar.update(1)

                # 2️⃣  Binary / model files — stream directly, no compression
                for file_path, arcname in files_stored:
                    zipf.write(file_path, arcname,
                               compress_type=zipfile.ZIP_STORED)
                    pbar.update(1)

                # 3️⃣  Compressible files — parallel deflation across CPU cores
                if files_parallel:
                    with ProcessPoolExecutor(max_workers=workers) as executor:
                        future_map = {
                            executor.submit(_compress_file_to_bytes, args): args
                            for args in files_parallel
                        }
                        for future in as_completed(future_map):
                            arcname, raw, _, err = future.result()
                            if err:
                                print(f"\n⚠️  Could not compress {arcname}: {err}")
                            else:
                                import io
                                mem = io.BytesIO(raw)
                                with zipfile.ZipFile(mem, "r") as src:
                                    for entry in src.infolist():
                                        zipf.writestr(entry, src.read(entry))
                            pbar.update(1)

        # ── Summary ───────────────────────────────────────────────────────────
        elapsed     = time.time() - start_time
        zip_size_mb = zip_filepath.stat().st_size / (1024 * 1024)
        ratio       = (1 - zip_size_mb / folder_size_mb) * 100 if folder_size_mb else 0

        print(f"\n✅  Created       : {zip_filepath}")
        print(f"✅  Original size : {folder_size_mb:.2f} MB")
        print(f"✅  Zip size      : {zip_size_mb:.2f} MB")
        print(f"✅  Space saved   : {ratio:.1f}%")
        print(f"⏱️   Time taken    : {elapsed:.2f} seconds")

    except Exception as e:
        print(f"Error creating zip file: {e}")
        raise


# ── Hardcoded project paths ───────────────────────────────────────────────────
PATH_QUANTIZED = "/home/nikhilsg/nikhilsg/sarvam-ai-30b-llm-quantization/research/quantized_models"
PATH_OUTPUTS   = "/home/nikhilsg/nikhilsg/sarvam-ai-30b-llm-quantization/research/outputs"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zip quantization project folders.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python zip.py --q            → quantized_models.zip\n"
            "  python zip.py --o            → outputs.zip\n"
            "  python zip.py --q --o        → both zips\n"
            "  python zip.py --q --q.st     → quantized_models_Nst.zip  (no .safetensors)\n"
            "  python zip.py --q --o --q.st → quantized_models_Nst.zip + outputs.zip\n"
        ),
    )
    parser.add_argument(
        "--q",
        action="store_true",
        help=f"Zip the quantized_models folder:\n  {PATH_QUANTIZED}",
    )
    parser.add_argument(
        "--o",
        action="store_true",
        help=f"Zip the outputs folder:\n  {PATH_OUTPUTS}",
    )
    parser.add_argument(
        "--q.st",
        dest="q_st",
        action="store_true",
        help=(
            "Skip all .safetensors files when zipping quantized_models.\n"
            "Changes output filename → quantized_models_Nst.zip\n"
            "Must be used together with --q."
        ),
    )

    args = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    if not args.q and not args.o:
        parser.error("Please specify at least one flag: --q, --o, or both.")
    if args.q_st and not args.q:
        parser.error("--q.st requires --q (it only applies to quantized_models).")

    # ── Build target list ─────────────────────────────────────────────────────
    targets = []

    if args.q:
        if args.q_st:
            # Skips .safetensors → suffix _Nst in filename
            targets.append((
                "quantized_models",
                PATH_QUANTIZED,
                {".safetensors"},
                "quantized_models_Nst.zip",   # ← custom name
            ))
        else:
            targets.append((
                "quantized_models",
                PATH_QUANTIZED,
                set(),
                "quantized_models.zip",        # ← default name
            ))

    if args.o:
        targets.append((
            "outputs",
            PATH_OUTPUTS,
            set(),
            "outputs.zip",
        ))

    # ── Run ───────────────────────────────────────────────────────────────────
    grand_start = time.time()

    for label, path, skip_ext, out_name in targets:
        print(f"\n{'═' * 60}")
        print(f"  📦  Target   : {label}")
        print(f"  📂  Path     : {path}")
        print(f"  💾  Output   : {out_name}")
        if skip_ext:
            print(f"  ⏭️   Skipping : {', '.join(sorted(skip_ext))}")
        print(f"{'═' * 60}")
        folder_to_zip(path, workers=None, skip_extensions=skip_ext, zip_name=out_name)

    if len(targets) > 1:
        print(f"\n{'═' * 60}")
        print(f"🏁  All done in {time.time() - grand_start:.2f} seconds total.")
        print(f"{'═' * 60}")

# import zipfile
# import sys
# import os

# def create_zip(target_path):
#     # Remove trailing slashes if any
#     target_path = target_path.rstrip(os.sep)
    
#     if not os.path.exists(target_path):
#         print(f"Error: {target_path} not found.")
#         return

#     # Use the target name for the zip name
#     zip_name = f"{target_path}.zip"

#     with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         if os.path.isdir(target_path):
#             # Walk through to maintain structure
#             for root, _, files in os.walk(target_path):
#                 for file in files:
#                     full_path = os.path.join(root, file)
#                     # relpath keeps the folder name and internal structure
#                     arc_path = os.path.relpath(full_path, os.path.dirname(target_path))
#                     zipf.write(full_path, arc_path)
#             print(f"Folder '{target_path}' zipped as '{zip_name}'")
            
#         else:
#             # For a single file
#             zipf.write(target_path, os.path.basename(target_path))
#             print(f"File '{target_path}' zipped as '{zip_name}'")

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python zip.py <folder_or_file_path>")
#     else:
#         # Takes the path provided at run command
#         create_zip(sys.argv[1])

