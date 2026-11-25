#!/usr/bin/env bash
# mpv-NFUA On-Demand RIFE Model Manager & Patcher (Linux)
# Interactive checkmark picker to install/remove optional RIFE models (v4.15-lite .. v4.26),
# patches rife.lua / input.conf on top of existing installs, verifies SHA256 checksums,
# and optionally pre-compiles TensorRT engines & calibrates per-model presets.
set -euo pipefail

DEFAULT_BASE_URL="https://github.com/NoobFromUA-YT/mpv-NFUA/releases/download/models-v1"
REPO_RAW_URL="https://raw.githubusercontent.com/NoobFromUA-YT/mpv-NFUA/main/source"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTALL_DIR=""
BASE_URL="$DEFAULT_BASE_URL"
PATCH_SOURCE=""
CLI_INSTALL=""
CLI_REMOVE=""
SET_DEFAULT=""
DO_LIST=0
DO_COMPILE=1
DO_PATCH=1
NO_PROMPT=0

# Ordered catalog of optional models (v4.6 is always built-in).
# Columns: id|label|onnx|align|bytes|sha256|description
MODEL_DEFS=(
    "v4.15_lite|RIFE v4.15-lite|rife_v4.15_lite.onnx|32|5285956|c6c5cd4402337cbf50a903a11fa07aa0842658622aeb9dcb3f3cd4dca255e729|5.0 MB | Fastest optional (~108 FPS @ 1080p, sharp edges)"
    "v4.22_lite|RIFE v4.22-lite|rife_v4.22_lite.onnx|32|9785548|e698112c6e336709b09a11e6b77bbcae4857c9fcd960507c3790208a89aca221|9.3 MB | Best real-time balance (~95 FPS @ 1080p)"
    "v4.25_lite|RIFE v4.25-lite|rife_v4.25_lite.onnx|128|11286011|083c9e1837f4656aa3ea3085c72a19613c449edeaf65f6df10fa6636d049e042|10.8 MB | Fast v4.25 lite (~110 FPS @ 1080p, mod128)"
    "v4.25|RIFE v4.25|rife_v4.25.onnx|32|11395540|0c749e1eccf4dedcecebfc2159ed676aa5553e985e76a55a9d252c1da9d98e1f|10.9 MB | Community favorite for anime/line art (~80 FPS)"
    "v4.25_heavy|RIFE v4.25-heavy|rife_v4.25_heavy.onnx|64|43359465|0405ca162de4f5915d6ce0cc0229d3d2e1aa90e7b76cd815f91e83397112321c|41.4 MB | Top quality flagship (~95 FPS @ 1080p on RTX 4070 Ti)"
    "v4.26|RIFE v4.26|rife_v4.26.onnx|64|11395540|7033af8175b0cb521d906206c819917ef03a84dcc310bfe193a48a0c78ff3091|10.9 MB | Latest official Practical-RIFE (~87 FPS @ 1080p)"
)

usage() {
    cat <<'EOF'
Usage: add-rife-models.sh [options]

Interactive mode (default on TTY): opens a checkmark selector to install or
remove optional RIFE models on top of an existing mpv-NFUA install, patches
scripts/rife.lua + input.conf for Alt+m / Alt+M model rolling & persistence,
and compiles TensorRT engines for newly added models.

Options:
  --dir <path>          mpv-NFUA install directory (auto-detected by default)
  --base-url <url|dir>  Download base URL or local directory for model ONNX files
  --patch-source <dir>  Local directory containing portable_config/scripts/rife.lua
  --install <list>      Comma-separated model IDs to install (e.g. v4.25,v4.26 or all)
  --remove <list>       Comma-separated optional model IDs to remove (or all)
  --set-default <id>    Save active startup model to script-opts/rife-state.conf
  --list                Show catalog and installed status, then exit
  --no-compile          Skip TensorRT engine compilation & preset calibration
  --no-patch            Do not patch rife.lua / input.conf
  --no-prompt           Non-interactive mode
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)          INSTALL_DIR="${2:?Missing value for --dir}"; shift 2 ;;
        --base-url)     BASE_URL="${2:?Missing value for --base-url}"; shift 2 ;;
        --patch-source) PATCH_SOURCE="${2:?Missing value for --patch-source}"; shift 2 ;;
        --install)      CLI_INSTALL="${2:?Missing value for --install}"; shift 2 ;;
        --remove)       CLI_REMOVE="${2:?Missing value for --remove}"; shift 2 ;;
        --set-default)  SET_DEFAULT="${2:?Missing value for --set-default}"; shift 2 ;;
        --list)         DO_LIST=1; shift ;;
        --no-compile)   DO_COMPILE=0; shift ;;
        --no-patch)     DO_PATCH=0; shift ;;
        --no-prompt)    NO_PROMPT=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -t 1 ]]; then
    C_CYAN=$'\033[1;36m'
    C_GREEN=$'\033[1;32m'
    C_YELLOW=$'\033[1;33m'
    C_RED=$'\033[1;31m'
    C_BOLD=$'\033[1m'
    C_DIM=$'\033[2m'
    C_REV=$'\033[7m'
    C_RESET=$'\033[0m'
else
    C_CYAN="" C_GREEN="" C_YELLOW="" C_RED="" C_BOLD="" C_DIM="" C_REV="" C_RESET=""
fi

step() { printf "\n%s==> %s%s\n" "$C_CYAN" "$*" "$C_RESET"; }
warn() { printf "%sWarning: %s%s\n" "$C_YELLOW" "$*" "$C_RESET" >&2; }
die()  { printf "%sError: %s%s\n" "$C_RED" "$*" "$C_RESET" >&2; exit 1; }

resolve_install_dir() {
    if [[ -n "$INSTALL_DIR" ]]; then
        INSTALL_DIR="$(realpath -m "$INSTALL_DIR")"
        return
    fi
    if [[ -d "$SCRIPT_DIR/portable_config/rife" || -x "$SCRIPT_DIR/bin/mpv" ]]; then
        INSTALL_DIR="$SCRIPT_DIR"
    elif [[ -d "$HOME/.local/share/mpv-nfua/portable_config" ]]; then
        INSTALL_DIR="$HOME/.local/share/mpv-nfua"
    elif [[ -d "$SCRIPT_DIR/../portable_config/rife" ]]; then
        INSTALL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    else
        INSTALL_DIR="$HOME/.local/share/mpv-nfua"
    fi
}

resolve_install_dir
RIFE_DIR="$INSTALL_DIR/portable_config/rife"
SCRIPTS_DIR="$INSTALL_DIR/portable_config/scripts"
OPTS_DIR="$INSTALL_DIR/portable_config/script-opts"
INPUT_CONF="$INSTALL_DIR/portable_config/input.conf"
STATE_FILE="$OPTS_DIR/rife-state.conf"

[[ -d "$INSTALL_DIR/portable_config" ]] || die "mpv-NFUA install not found at '$INSTALL_DIR' (pass --dir <path>)."

N_MODELS=${#MODEL_DEFS[@]}
declare -a M_ID M_LABEL M_ONNX M_ALIGN M_BYTES M_SHA M_DESC M_INSTALLED M_CHECKED

for ((i=0; i<N_MODELS; i++)); do
    IFS='|' read -r id label onnx align bytes sha desc <<<"${MODEL_DEFS[$i]}"
    M_ID[$i]="$id"
    M_LABEL[$i]="$label"
    M_ONNX[$i]="$onnx"
    M_ALIGN[$i]="$align"
    M_BYTES[$i]="$bytes"
    M_SHA[$i]="$sha"
    M_DESC[$i]="$desc"
    if [[ -f "$RIFE_DIR/$onnx" ]]; then
        M_INSTALLED[$i]=1
        M_CHECKED[$i]=1
    else
        M_INSTALLED[$i]=0
        M_CHECKED[$i]=0
    fi
done

find_idx() {
    local q="${1,,}"
    q="${q#rife_}"
    q="${q//-/_}"
    for ((i=0; i<N_MODELS; i++)); do
        if [[ "${M_ID[$i],,}" == "$q" ]]; then
            echo "$i"
            return 0
        fi
    done
    return 1
}

read_saved_state() {
    if [[ -f "$STATE_FILE" ]]; then
        awk -F= '/^[[:space:]]*model[[:space:]]*=/ { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2 }' "$STATE_FILE" | tail -n 1
    fi
}

write_saved_state() {
    local mid="$1"
    mkdir -p "$OPTS_DIR"
    cat >"$STATE_FILE" <<EOF
# Saved active RIFE model (updated by scripts/rife.lua on Alt+m / Alt+M).
model=$mid
EOF
}

if [[ "$DO_LIST" -eq 1 ]]; then
    saved="$(read_saved_state || true)"
    printf "%sInstalled RIFE models in %s:%s\n" "$C_BOLD" "$RIFE_DIR" "$C_RESET"
    printf "  [x] %-12s %-18s %s\n" "v4.6" "RIFE v4.6" "(built-in default, full + half flow)"
    for ((i=0; i<N_MODELS; i++)); do
        mark=" "
        tag=""
        if [[ "${M_INSTALLED[$i]}" -eq 1 ]]; then
            mark="x"
            tag=" ${C_GREEN}[installed]${C_RESET}"
        fi
        printf "  [%s] %-12s %-18s %s%b\n" "$mark" "${M_ID[$i]}" "${M_LABEL[$i]}" "${M_DESC[$i]}" "$tag"
    done
    printf "\nStartup model (rife-state.conf): %s\n" "${saved:-v4.6 (default)}"
    exit 0
fi

# Interactive checkmark TUI (pure bash, reads from /dev/tty)
run_tui() {
    local cur=0
    local key rest
    local stty_saved
    stty_saved="$(stty -g </dev/tty)"
    cleanup_tui() {
        printf "\033[?25h" >/dev/tty 2>/dev/null || true
        stty "$stty_saved" </dev/tty 2>/dev/null || true
    }
    trap cleanup_tui EXIT INT TERM
    printf "\033[?25l" >/dev/tty

    while true; do
        printf "\033[H\033[J" >/dev/tty
        printf "%smpv-NFUA — On-Demand RIFE Model Manager & Patcher%s\n" "$C_CYAN" "$C_RESET" >/dev/tty
        printf "%sInstall folder: %s%s\n\n" "$C_DIM" "$INSTALL_DIR" "$C_RESET" >/dev/tty
        printf "  %s[x] %-17s %s%s\n" "$C_GREEN" "RIFE v4.6" "Built-in base model (always kept, full + half flow)" "$C_RESET" >/dev/tty
        printf "  ──────────────────────────────────────────────────────────────────────────\n" >/dev/tty
        for ((i=0; i<N_MODELS; i++)); do
            local mark=" " prefix="  "
            local status=""
            [[ "${M_CHECKED[$i]}" -eq 1 ]] && mark="x"
            if [[ "${M_INSTALLED[$i]}" -eq 1 && "${M_CHECKED[$i]}" -eq 1 ]]; then
                status=" ${C_GREEN}(installed)${C_RESET}"
            elif [[ "${M_INSTALLED[$i]}" -eq 1 && "${M_CHECKED[$i]}" -eq 0 ]]; then
                status=" ${C_RED}(will remove)${C_RESET}"
            elif [[ "${M_INSTALLED[$i]}" -eq 0 && "${M_CHECKED[$i]}" -eq 1 ]]; then
                status=" ${C_CYAN}(will install)${C_RESET}"
            fi
            if [[ "$i" -eq "$cur" ]]; then
                printf " %s>%s %s[%s] %d. %-16s %s%b\n" "$C_CYAN" "$C_RESET" "$C_BOLD" "$mark" "$((i+1))" "${M_LABEL[$i]}" "${M_DESC[$i]}" "$status$C_RESET" >/dev/tty
            else
                printf "   [%s] %d. %-16s %s%b\n" "$mark" "$((i+1))" "${M_LABEL[$i]}" "${M_DESC[$i]}" "$status" >/dev/tty
            fi
        done
        printf "\n  %sControls:%s  Up/Down move  |  Space or 1-%d toggle  |  a all  |  n none  |  Enter apply  |  q quit\n" \
            "$C_BOLD" "$C_RESET" "$N_MODELS" >/dev/tty

        IFS= read -rsn1 key </dev/tty || break
        if [[ "$key" == $'\x1b' ]]; then
            IFS= read -rsn2 -t 0.1 rest </dev/tty || rest=""
            case "$rest" in
                "[A") cur=$(( (cur - 1 + N_MODELS) % N_MODELS )) ;;
                "[B") cur=$(( (cur + 1) % N_MODELS )) ;;
            esac
        elif [[ "$key" == " " ]]; then
            M_CHECKED[$cur]=$(( 1 - M_CHECKED[$cur] ))
        elif [[ "$key" == "k" || "$key" == "K" ]]; then
            cur=$(( (cur - 1 + N_MODELS) % N_MODELS ))
        elif [[ "$key" == "j" || "$key" == "J" ]]; then
            cur=$(( (cur + 1) % N_MODELS ))
        elif [[ "$key" =~ ^[1-6]$ ]]; then
            local idx=$(( key - 1 ))
            cur=$idx
            M_CHECKED[$idx]=$(( 1 - M_CHECKED[$idx] ))
        elif [[ "$key" == "a" || "$key" == "A" ]]; then
            for ((i=0; i<N_MODELS; i++)); do M_CHECKED[$i]=1; done
        elif [[ "$key" == "n" || "$key" == "N" ]]; then
            for ((i=0; i<N_MODELS; i++)); do M_CHECKED[$i]=0; done
        elif [[ "$key" == "" ]]; then
            break
        elif [[ "$key" == "q" || "$key" == "Q" ]]; then
            cleanup_tui
            trap - EXIT INT TERM
            printf "\nCancelled.\n"
            exit 0
        fi
    done

    cleanup_tui
    trap - EXIT INT TERM
    printf "\n" >/dev/tty
}

if [[ -n "$CLI_INSTALL" || -n "$CLI_REMOVE" ]]; then
    if [[ -n "$CLI_INSTALL" ]]; then
        if [[ "$CLI_INSTALL" == "all" ]]; then
            for ((i=0; i<N_MODELS; i++)); do M_CHECKED[$i]=1; done
        else
            IFS=',' read -ra toks <<<"$CLI_INSTALL"
            for tok in "${toks[@]}"; do
                tok="${tok// /}"
                [[ -z "$tok" ]] && continue
                idx="$(find_idx "$tok")" || die "Unknown model ID '$tok' (use --list to see valid IDs)."
                M_CHECKED[$idx]=1
            done
        fi
    fi
    if [[ -n "$CLI_REMOVE" ]]; then
        if [[ "$CLI_REMOVE" == "all" ]]; then
            for ((i=0; i<N_MODELS; i++)); do M_CHECKED[$i]=0; done
        else
            IFS=',' read -ra toks <<<"$CLI_REMOVE"
            for tok in "${toks[@]}"; do
                tok="${tok// /}"
                [[ -z "$tok" ]] && continue
                idx="$(find_idx "$tok")" || die "Unknown model ID '$tok' (use --list to see valid IDs)."
                M_CHECKED[$idx]=0
            done
        fi
    fi
elif [[ "$NO_PROMPT" -eq 0 && -e /dev/tty && -t 0 ]]; then
    run_tui
elif [[ -z "$SET_DEFAULT" ]]; then
    die "Nothing specified in non-interactive mode (pass --install, --remove, --set-default, or --list)."
fi

fetch_item() {
    local src="$1" dst="$2" label="$3"
    if [[ "$src" == http://* || "$src" == https://* ]]; then
        command -v curl >/dev/null 2>&1 || die "curl is required for downloading $label"
        curl -fL --retry 3 --retry-delay 2 --progress-bar -o "$dst" "$src"
    else
        [[ -f "$src" ]] || return 1
        cp -f "$src" "$dst"
    fi
}

# 1. Patcher on top: update scripts/rife.lua, input.conf, and rife.conf cleanly
patch_player() {
    step "Patching player scripts & hotkeys (Alt+m / Alt+M + persistence)"
    mkdir -p "$SCRIPTS_DIR" "$OPTS_DIR"

    local src_lua=""
    local cand_roots=()
    [[ -n "$PATCH_SOURCE" ]] && cand_roots+=("$PATCH_SOURCE")
    cand_roots+=("$SCRIPT_DIR/source" "$SCRIPT_DIR")
    for root in "${cand_roots[@]}"; do
        if [[ -f "$root/portable_config/scripts/rife.lua" ]] && grep -q "rife-model" "$root/portable_config/scripts/rife.lua" && ! grep -q "align=%d" "$root/portable_config/scripts/rife.lua"; then
            src_lua="$root/portable_config/scripts/rife.lua"
            break
        fi
    done

    local lua_target="$SCRIPTS_DIR/rife.lua"
    if [[ -f "$lua_target" ]] && grep -q "rife-model" "$lua_target" && ! grep -q "align=%d" "$lua_target"; then
        printf "  scripts/rife.lua already supports multi-model rolling & persistence.\n"
    else
        local tmp_lua="$lua_target.tmp.$$"
        if [[ -n "$src_lua" ]]; then
            cp -f "$src_lua" "$tmp_lua"
            mv -f "$tmp_lua" "$lua_target"
            printf "  Patched scripts/rife.lua from %s\n" "$src_lua"
        elif fetch_item "$REPO_RAW_URL/portable_config/scripts/rife.lua" "$tmp_lua" "rife.lua" && grep -q "rife-model" "$tmp_lua" && ! grep -q "align=%d" "$tmp_lua"; then
            mv -f "$tmp_lua" "$lua_target"
            printf "  Updated scripts/rife.lua from GitHub.\n"
        else
            rm -f "$tmp_lua"
            warn "Could not download updated scripts/rife.lua (check network or pass --patch-source)."
        fi
    fi

    if [[ ! -f "$INPUT_CONF" ]]; then
        touch "$INPUT_CONF"
    fi
    if grep -q "script-message rife-model" "$INPUT_CONF"; then
        printf "  input.conf already binds Alt+m / Alt+M.\n"
    else
        python3 - "$INPUT_CONF" <<'PY'
import sys
p = sys.argv[1]
txt = open(p, "r", encoding="utf-8").read()
block = (
    "# Step through installed RIFE models (Alt+m next, Alt+M previous). Saved across restarts.\n"
    "Alt+m script-message rife-model cycle\n"
    "Alt+M script-message rife-model prev\n\n"
)
marker = "# Measure this PC"
if marker in txt:
    txt = txt.replace(marker, block + marker, 1)
elif "Alt+6 script-message rife 6\n" in txt:
    txt = txt.replace("Alt+6 script-message rife 6\n", "Alt+6 script-message rife 6\n\n" + block, 1)
else:
    if txt and not txt.endswith("\n"):
        txt += "\n"
    txt += "\n" + block
open(p, "w", encoding="utf-8").write(txt)
PY
        printf "  Added Alt+m / Alt+M bindings to portable_config/input.conf\n"
    fi

    local rife_conf="$OPTS_DIR/rife.conf"
    if [[ -f "$rife_conf" ]] && ! grep -q "model=" "$rife_conf"; then
        python3 - "$rife_conf" <<'PY'
import sys
p = sys.argv[1]
txt = open(p, "r", encoding="utf-8").read()
add = (
    "# Force a startup RIFE model (empty = remember last model picked with Alt+m / Alt+M).\n"
    "# Extra models can be added on demand with add-rife-models.sh / Add-RIFE-Models.bat.\n"
    "#model=v4.6\n\n"
)
marker = "# The screen's refresh rate"
if marker in txt:
    txt = txt.replace(marker, add + marker, 1)
else:
    txt = add + txt
open(p, "w", encoding="utf-8").write(txt)
PY
        printf "  Updated portable_config/script-opts/rife.conf comments.\n"
    fi
}

[[ "$DO_PATCH" -eq 1 ]] && patch_player

# 2. Remove unchecked models + their TensorRT engine caches + presets
removed_any=()
for ((i=0; i<N_MODELS; i++)); do
    if [[ "${M_INSTALLED[$i]}" -eq 1 && "${M_CHECKED[$i]}" -eq 0 ]]; then
        id="${M_ID[$i]}"
        onnx="${M_ONNX[$i]}"
        if [[ "${#removed_any[@]}" -eq 0 ]]; then
            step "Removing unchecked models & freeing engine cache"
        fi
        rm -f "$RIFE_DIR/$onnx" "$RIFE_DIR/$onnx".*.engine "$OPTS_DIR/rife-preset-${id}.conf"
        printf "  Removed %s (%s + cached engines)\n" "${M_LABEL[$i]}" "$onnx"
        M_INSTALLED[$i]=0
        removed_any+=("$id")
    fi
done

saved_now="$(read_saved_state || true)"
for rid in "${removed_any[@]}"; do
    if [[ "$saved_now" == "$rid" ]]; then
        write_saved_state "v4.6"
        printf "  Reset saved startup model from %s back to v4.6.\n" "$rid"
    fi
done

# 3. Install newly checked models with SHA256 verification
added_idxs=()
for ((i=0; i<N_MODELS; i++)); do
    if [[ "${M_INSTALLED[$i]}" -eq 0 && "${M_CHECKED[$i]}" -eq 1 ]]; then
        if [[ "${#added_idxs[@]}" -eq 0 ]]; then
            step "Downloading checked RIFE models"
        fi
        mkdir -p "$RIFE_DIR"
        onnx="${M_ONNX[$i]}"
        dst="$RIFE_DIR/$onnx"
        tmp="$dst.part.$$"
        url="${BASE_URL%/}/$onnx"
        printf "  [%s] Fetching %s...\n" "${M_LABEL[$i]}" "$url"
        if ! fetch_item "$url" "$tmp" "$onnx"; then
            rm -f "$tmp"
            die "Failed to fetch $onnx from $url"
        fi
        actual_sha="$(sha256sum "$tmp" | awk '{print $1}')"
        if [[ "$actual_sha" != "${M_SHA[$i]}" ]]; then
            rm -f "$tmp"
            die "SHA256 mismatch for $onnx (expected ${M_SHA[$i]}, got $actual_sha)"
        fi
        mv -f "$tmp" "$dst"
        printf "  %s[SHA256 OK]%s Saved %s (%s bytes)\n" "$C_GREEN" "$C_RESET" "$onnx" "${M_BYTES[$i]}"
        M_INSTALLED[$i]=1
        added_idxs+=("$i")
    fi
done

# 4. Pre-compile TensorRT engines & calibrate presets for newly added models
detect_screen_fps() {
    local fps=""
    if command -v xrandr >/dev/null 2>&1; then
        fps="$(xrandr --current 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)?\*' | head -n 1 | tr -d '*' || true)"
    fi
    if [[ -n "$fps" ]]; then
        python3 -c "import sys; print(int(round(float(sys.argv[1]))))" "$fps" 2>/dev/null || echo "160"
    elif [[ -f "$OPTS_DIR/rife-preset.conf" ]]; then
        awk -F= '/^screen_fps=/ { print $2; exit }' "$OPTS_DIR/rife-preset.conf"
    else
        echo "160"
    fi
}

bench_bin=""
for cand in "$INSTALL_DIR/bin/rife-bench" "$INSTALL_DIR/rife-bench" "$INSTALL_DIR/build/video/filter/rife/rife-bench"; do
    if [[ -x "$cand" ]]; then
        bench_bin="$cand"
        break
    fi
done

if [[ "$DO_COMPILE" -eq 1 && "${#added_idxs[@]}" -gt 0 ]]; then
    if [[ -z "$bench_bin" ]]; then
        warn "rife-bench not found; TensorRT engines will compile on first playback instead."
    else
        step "Compiling TensorRT engines & calibrating newly added models"
        screen_fps="$(detect_screen_fps)"
        screen_fps="${screen_fps:-160}"
        for idx in "${added_idxs[@]}"; do
            id="${M_ID[$idx]}"
            label="${M_LABEL[$idx]}"
            onnx="${M_ONNX[$idx]}"
            align="${M_ALIGN[$idx]}"
            printf "  Compiling & timing %s (align=%s)...\n" "$label" "$align"
            set +e
            out="$("$bench_bin" --model "$RIFE_DIR/$onnx" --align "$align" --sizes "1280x720,1920x1080,2560x1440,3840x2160" --runs 10 2>&1)"
            rc=$?
            set -e
            if [[ "$rc" -ne 0 ]] && grep -q "unknown option --align" <<<"$out"; then
                warn "Installed rife-bench lacks --align; skipping offline calibration (first playback will auto-compile $label)."
                break
            elif [[ "$rc" -ne 0 ]]; then
                warn "rife-bench failed for $label (exit $rc): $out"
                continue
            fi
            printf "%s\n" "$out" | grep -E "^(gpu|result) " | sed 's/^/    /'
            gpu="$(awk '/^gpu / { sub(/^gpu /, ""); print; exit }' <<<"$out")"
            pts="$(awk '/^result full / { printf "%s%s:%s", sep, $4, $5; sep="," }' <<<"$out")"
            if [[ -n "$pts" ]]; then
                mkdir -p "$OPTS_DIR"
                stamp="$(date '+%Y-%m-%d %H:%M')"
                cat >"$OPTS_DIR/rife-preset-${id}.conf" <<EOF
# Written by add-rife-models.sh on $stamp.
# Model: $label ($onnx)
# Alt+9 in the player measures again; delete this file to measure at the next start.
# A setting in rife.conf wins over the same one here.

model=$id
measured=$stamp
gpu=${gpu:-NVIDIA GPU}
screen_fps=$screen_fps

# GPU time for one interpolated frame, as tensor megapixels:ms (sizes 1280x720,1920x1080,2560x1440,3840x2160).
full=$pts
EOF
                printf "    Wrote script-opts/rife-preset-%s.conf\n" "$id"
            fi
        done
    fi
fi

# 5. Save startup model selection
if [[ -n "$SET_DEFAULT" ]]; then
    if [[ "$SET_DEFAULT" == "v4.6" ]]; then
        write_saved_state "v4.6"
        printf "\nStartup model saved: RIFE v4.6 (rife-state.conf)\n"
    else
        didx="$(find_idx "$SET_DEFAULT")" || die "Unknown model ID for --set-default: $SET_DEFAULT"
        [[ "${M_INSTALLED[$didx]}" -eq 1 ]] || die "Cannot set $SET_DEFAULT as default because it is not installed."
        write_saved_state "${M_ID[$didx]}"
        printf "\nStartup model saved: %s (rife-state.conf)\n" "${M_LABEL[$didx]}"
    fi
elif [[ "${#added_idxs[@]}" -gt 0 && "$NO_PROMPT" -eq 0 && -e /dev/tty && -t 0 ]]; then
    last_added_idx="${added_idxs[$((${#added_idxs[@]} - 1))]}"
    read -r -p $'\nSwitch startup model to '"${M_LABEL[$last_added_idx]}"'? [Y/n]: ' ans </dev/tty || true
    if [[ -z "$ans" || "$ans" =~ ^[Yy] ]]; then
        write_saved_state "${M_ID[$last_added_idx]}"
        printf "  Startup model saved: %s\n" "${M_LABEL[$last_added_idx]}"
    fi
fi

step "Done!"
printf "  In mpv-NFUA, press %sAlt+m%s (next) or %sAlt+M%s (previous) to roll between installed RIFE models.\n" "$C_BOLD" "$C_RESET" "$C_BOLD" "$C_RESET"
printf "  Your last chosen model is automatically saved to portable_config/script-opts/rife-state.conf.\n"
