"""E1-E12 + variants runner (reduced budget: 15% data / 8 epochs each).
Usage: python experiments/run_all.py   (run from repo root)
Reused, not re-run: E3_d03 as E4-off/E6-256/E7-64/E11-mid; Phase 7 for E8;
Phase 5 variants for E9; Phases 4-6 for E10.
"""
import subprocess, sys

RUNS = [
    ["--config", "exp", "--tag", "E12_gpu", "--subset", "0.05", "--epochs",
     "2", "--batch", "64", "--stride", "2"],
    ["--config", "exp", "--tag", "var_many2many", "--model", "many2many"],
    ["--config", "exp", "--tag", "var_stateful", "--model", "stateful"],
    ["--config", "exp", "--tag", "var_cnn", "--model", "cnn"],
    * [["--config", "exp", "--tag", f"E1_{o}", "--opt", o]
       for o in ["sgd", "sgd_mom", "nag", "adagrad", "adadelta", "rmsprop",
                 "adam"]],
    ["--config", "exp", "--tag", "E2_clip_on", "--lr", "0.01", "--clip",
     "1.0"],
    ["--config", "exp", "--tag", "E2_clip_off", "--lr", "0.01", "--clip",
     "0"],
    ["--config", "exp", "--tag", "E3_d0", "--dropout", "0"],
    ["--config", "exp", "--tag", "E3_d03", "--dropout", "0.3"],
    ["--config", "exp", "--tag", "E3_d05", "--dropout", "0.5"],
    ["--config", "exp", "--tag", "E4_bn", "--batchnorm"],
    ["--config", "exp", "--tag", "E6_u128", "--units", "128"],
    ["--config", "exp", "--tag", "E6_u512", "--units", "512"],
    ["--config", "exp", "--tag", "E7_s32", "--seq-len", "32"],
    ["--config", "exp", "--tag", "E7_s100", "--seq-len", "100"],
    ["--config", "exp", "--tag", "E11_b32_lr1e-3", "--batch", "32", "--lr",
     "0.001"],
    ["--config", "exp", "--tag", "E11_b64_lr1e-3", "--batch", "64", "--lr",
     "0.001"],
    ["--config", "exp", "--tag", "E11_b32_lr3e-3", "--batch", "32", "--lr",
     "0.003"],
    ["--config", "exp", "--tag", "E11_b32_lr1e-2", "--batch", "32", "--lr",
     "0.01"],
    ["--config", "exp", "--tag", "E11_b64_lr1e-2", "--batch", "64", "--lr",
     "0.01"],
]

if __name__ == "__main__":
    log = open("outputs/logs/experiments_all.log", "a", encoding="utf-8")
    for cmd in RUNS:
        tag = cmd[cmd.index("--tag") + 1]
        print(f"=== {tag} ===", flush=True)
        log.write(f"=== {tag} ===\n")
        log.flush()
        p = subprocess.run([sys.executable, "src/train.py", *cmd],
                           capture_output=True, text=True)
        log.write(p.stdout)
        log.write(p.stderr)
        log.flush()
        if p.returncode != 0:
            print(f"{tag} FAILED rc={p.returncode}\n{p.stderr[-2000:]}")
            break
        print(p.stdout.strip().split("\n")[-1])
    print("ALL_EXPERIMENTS_DONE")
    log.write("ALL_EXPERIMENTS_DONE\n")
