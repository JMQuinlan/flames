"""Bundle an analysis script's results into one .tar.gz for transfer (e.g. INCLINE -> laptop).

bundle(figdir, stem, plotdirs) writes <figdir>/<stem>_results.tar.gz containing
  * every file in figdir whose name starts with <stem> (figures .png/.eps, .gif, cached .npz/.csv, ...),
    except earlier archives, plus the GIF frame folders <stem>_gif_frames/ (EPS frames);
  * for each plot directory: its thermo.dat and metadata (small; plotfiles are NOT included).
Disable with the environment variable MARMPAPER_NO_BUNDLE=1.
"""
import os
import tarfile


def bundle(figdir, stem, plotdirs=()):
    if os.environ.get("MARMPAPER_NO_BUNDLE"):
        return None
    figdir = os.path.abspath(figdir)
    name = os.path.basename(stem) + "_results"
    out = os.path.join(figdir, name + ".tar.gz")
    files = []
    for f in sorted(os.listdir(figdir)):
        p = os.path.join(figdir, f)
        if os.path.isfile(p) and f.startswith(os.path.basename(stem)) and not f.endswith(".tar.gz"):
            files.append((p, os.path.join(name, f)))
    for f in sorted(os.listdir(figdir)):                 # GIF frame folders (<stem>_gif_frames/, EPS frames)
        p = os.path.join(figdir, f)
        if os.path.isdir(p) and f.startswith(os.path.basename(stem)) and f.endswith("_gif_frames"):
            for dp, _, fns in os.walk(p):
                for fn in sorted(fns):
                    files.append((os.path.join(dp, fn), os.path.join(name, os.path.relpath(os.path.join(dp, fn), figdir))))
    for d in plotdirs or ():
        if not d or not os.path.isdir(d):
            continue
        run = os.path.basename(os.path.normpath(d))
        for f in ("thermo.dat", "metadata"):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                files.append((p, os.path.join(name, "run_data", run, f)))
    if not files:
        return None
    tmp = out + ".part"
    with tarfile.open(tmp, "w:gz") as tar:
        for p, arc in files:
            tar.add(p, arcname=arc)
    os.replace(tmp, out)
    print("  bundled %d files -> %s (%.1f MB)" % (len(files), out, os.path.getsize(out) / 1e6))
    return out
