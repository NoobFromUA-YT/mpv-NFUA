@echo off
setlocal
set "MPV_NFUA_BAT_DIR=%~dp0"
set "PS1=%TEMP%\mpv_nfua_models_%RANDOM%_%RANDOM%.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$c=[IO.File]::ReadAllText('%~f0',[Text.Encoding]::UTF8); $m='###'+'MPV_NFUA_MODELS_PS1###'; $i=$c.IndexOf($m); if($i -lt 0){throw 'Payload marker not found.'}; [IO.File]::WriteAllText('%PS1%',$c.Substring($i+$m.Length),(New-Object Text.UTF8Encoding($false)))"
if errorlevel 1 (
    echo Failed to unpack model manager payload.
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
set "RC=%ERRORLEVEL%"
del /f /q "%PS1%" >nul 2>&1
echo %* | findstr /i "NoPrompt" >nul
if errorlevel 1 (
    echo.
    if not "%RC%"=="0" echo Model manager finished with error code %RC%.
    pause
)
exit /b %RC%
###MPV_NFUA_MODELS_PS1###
param(
    [string]$InstallDir = "",
    [string]$BaseUrl = "https://github.com/NoobFromUA-YT/mpv-NFUA/releases/download/models-v1",
    [string]$PatchSource = "",
    [string]$Install = "",
    [string]$Remove = "",
    [string]$SetDefault = "",
    [switch]$List,
    [switch]$NoCompile,
    [switch]$NoPatch,
    [switch]$NoPrompt
)

$ErrorActionPreference = "Stop"
$RepoRawUrl = "https://raw.githubusercontent.com/NoobFromUA-YT/mpv-NFUA/main/source"

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    if ([Enum]::GetNames([Net.SecurityProtocolType]) -contains "Tls13") {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls13
    }
} catch {}

$Catalog = @(
    [pscustomobject]@{ Id = "v4.15_lite";  Label = "RIFE v4.15-lite";  Onnx = "rife_v4.15_lite.onnx";  Align = 32;  Bytes = 5285956;  Sha256 = "c6c5cd4402337cbf50a903a11fa07aa0842658622aeb9dcb3f3cd4dca255e729"; Desc = "5.0 MB  | Fastest optional (~108 FPS @ 1080p, sharp edges)" }
    [pscustomobject]@{ Id = "v4.22_lite";  Label = "RIFE v4.22-lite";  Onnx = "rife_v4.22_lite.onnx";  Align = 32;  Bytes = 9785548;  Sha256 = "e698112c6e336709b09a11e6b77bbcae4857c9fcd960507c3790208a89aca221"; Desc = "9.3 MB  | Best real-time balance (~95 FPS @ 1080p)" }
    [pscustomobject]@{ Id = "v4.25_lite";  Label = "RIFE v4.25-lite";  Onnx = "rife_v4.25_lite.onnx";  Align = 128; Bytes = 11286011; Sha256 = "083c9e1837f4656aa3ea3085c72a19613c449edeaf65f6df10fa6636d049e042"; Desc = "10.8 MB | Fast v4.25 lite (~110 FPS @ 1080p, mod128)" }
    [pscustomobject]@{ Id = "v4.25";       Label = "RIFE v4.25";       Onnx = "rife_v4.25.onnx";       Align = 32;  Bytes = 11395540; Sha256 = "0c749e1eccf4dedcecebfc2159ed676aa5553e985e76a55a9d252c1da9d98e1f"; Desc = "10.9 MB | Community favorite for anime/line art (~80 FPS)" }
    [pscustomobject]@{ Id = "v4.25_heavy"; Label = "RIFE v4.25-heavy"; Onnx = "rife_v4.25_heavy.onnx"; Align = 64;  Bytes = 43359465; Sha256 = "0405ca162de4f5915d6ce0cc0229d3d2e1aa90e7b76cd815f91e83397112321c"; Desc = "41.4 MB | Top quality flagship (~95 FPS @ 1080p)" }
    [pscustomobject]@{ Id = "v4.26";       Label = "RIFE v4.26";       Onnx = "rife_v4.26.onnx";       Align = 64;  Bytes = 11395540; Sha256 = "7033af8175b0cb521d906206c819917ef03a84dcc310bfe193a48a0c78ff3091"; Desc = "10.9 MB | Latest official Practical-RIFE (~87 FPS)" }
)

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Resolve-InstallDir {
    if ($InstallDir) { return [IO.Path]::GetFullPath($InstallDir) }
    $batDir = if ($env:MPV_NFUA_BAT_DIR) { $env:MPV_NFUA_BAT_DIR.TrimEnd("\") } else { (Get-Location).Path }
    $candidates = @(
        $batDir,
        (Join-Path $env:LOCALAPPDATA "MPV-NFUA"),
        (Join-Path $batDir "..")
    )
    foreach ($cand in $candidates) {
        if ($cand -and (Test-Path (Join-Path $cand "portable_config"))) {
            return [IO.Path]::GetFullPath($cand)
        }
    }
    if ($NoPrompt) { throw "mpv-NFUA install folder not found. Pass -InstallDir <path>." }
    $chosen = Read-Host "mpv-NFUA install folder [$(Join-Path $env:LOCALAPPDATA 'MPV-NFUA')]"
    if (-not $chosen -or -not $chosen.Trim()) { return (Join-Path $env:LOCALAPPDATA "MPV-NFUA") }
    return [IO.Path]::GetFullPath($chosen.Trim().Trim('"'))
}

function Find-Model([string]$query) {
    if (-not $query) { return $null }
    $norm = ($query.Trim().ToLowerInvariant() -replace "^rife_", "" -replace "-", "_")
    foreach ($m in $Catalog) {
        if ($m.Id.ToLowerInvariant() -eq $norm) { return $m }
    }
    return $null
}

function Read-SavedState([string]$statePath) {
    if (-not (Test-Path $statePath)) { return "" }
    foreach ($line in Get-Content -LiteralPath $statePath -Encoding UTF8) {
        if ($line -match "^\s*model\s*=\s*(.+?)\s*$") { return $Matches[1].Trim() }
    }
    return ""
}

function Write-SavedState([string]$statePath, [string]$modelId) {
    $dir = Split-Path -Parent $statePath
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $content = "# Saved active RIFE model (updated by scripts/rife.lua on Alt+m / Alt+M).`r`nmodel=$modelId`r`n"
    [IO.File]::WriteAllText($statePath, $content, (New-Object Text.UTF8Encoding($false)))
}

function Download-OrCopy([string]$src, [string]$dst, [string]$label) {
    if ($src -match "^https?://") {
        $req = [Net.HttpWebRequest]::Create($src)
        $req.Method = "GET"
        $req.AllowAutoRedirect = $true
        $req.UserAgent = "MPV-NFUA-ModelManager/1.0"
        $resp = $req.GetResponse()
        try {
            [long]$total = $resp.ContentLength
            $stream = $resp.GetResponseStream()
            try {
                $fs = New-Object IO.FileStream($dst, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None, 262144)
                try {
                    $buf = New-Object byte[] 262144
                    [long]$done = 0
                    $lastPct = -10
                    while (($n = $stream.Read($buf, 0, $buf.Length)) -gt 0) {
                        $fs.Write($buf, 0, $n)
                        $done += $n
                        if ($total -gt 0) {
                            $pct = [int](($done * 100) / $total)
                            if ($pct -ge $lastPct + 5) {
                                $lastPct = $pct
                                Write-Progress -Activity "Downloading $label" -Status ("{0:N1} / {1:N1} MB ({2}%)" -f ($done / 1MB), ($total / 1MB), $pct) -PercentComplete $pct
                            }
                        }
                    }
                    Write-Progress -Activity "Downloading $label" -Completed
                } finally { $fs.Dispose() }
            } finally { $stream.Dispose() }
        } finally { $resp.Dispose() }
    } else {
        if (-not (Test-Path $src)) { throw "Source file not found: $src" }
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }
}

$InstallDir = Resolve-InstallDir
$PortableDir = Join-Path $InstallDir "portable_config"
$RifeDir = Join-Path $PortableDir "rife"
$ScriptsDir = Join-Path $PortableDir "scripts"
$OptsDir = Join-Path $PortableDir "script-opts"
$InputConf = Join-Path $PortableDir "input.conf"
$StateFile = Join-Path $OptsDir "rife-state.conf"

if (-not (Test-Path $PortableDir)) {
    throw "mpv-NFUA install not found at '$InstallDir' (missing portable_config\)."
}

$State = @{}
foreach ($m in $Catalog) {
    $isInst = Test-Path (Join-Path $RifeDir $m.Onnx)
    $State[$m.Id] = [pscustomobject]@{ Model = $m; Installed = $isInst; Checked = $isInst }
}

if ($List) {
    $saved = Read-SavedState $StateFile
    Write-Host "Installed RIFE models in $RifeDir`:" -ForegroundColor Green
    Write-Host "  [x] v4.6         RIFE v4.6          (built-in default, full + half flow)"
    foreach ($m in $Catalog) {
        $st = $State[$m.Id]
        $mark = if ($st.Installed) { "x" } else { " " }
        $tag = if ($st.Installed) { " [installed]" } else { "" }
        Write-Host ("  [{0}] {1,-12} {2,-18} {3}{4}" -f $mark, $m.Id, $m.Label, $m.Desc, $tag)
    }
    Write-Host ""
    Write-Host ("Startup model (rife-state.conf): {0}" -f $(if ($saved) { $saved } else { "v4.6 (default)" }))
    exit 0
}

$WantCompile = -not $NoCompile

function Show-GuiSelector([string]$currentDefault) {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [Windows.Forms.Application]::EnableVisualStyles()

    $form = New-Object Windows.Forms.Form
    $form.Text = "mpv-NFUA — RIFE Model Manager"
    $form.Size = New-Object Drawing.Size(660, 445)
    $form.StartPosition = "CenterScreen"
    $form.FormBorderStyle = "FixedDialog"
    $form.MaximizeBox = $false
    $form.MinimizeBox = $false
    $form.Font = New-Object Drawing.Font("Segoe UI", 9.5)

    $lblHead = New-Object Windows.Forms.Label
    $lblHead.Text = "Select optional RIFE AI models to install (uncheck any optional model to remove it):"
    $lblHead.Location = New-Object Drawing.Point(16, 14)
    $lblHead.Size = New-Object Drawing.Size(615, 22)
    $form.Controls.Add($lblHead)

    $lblBase = New-Object Windows.Forms.Label
    $lblBase.Text = "[x] RIFE v4.6 (Built-in default — always kept, full + half flow)"
    $lblBase.ForeColor = [Drawing.Color]::DarkGreen
    $lblBase.Location = New-Object Drawing.Point(18, 40)
    $lblBase.Size = New-Object Drawing.Size(615, 20)
    $form.Controls.Add($lblBase)

    $clb = New-Object Windows.Forms.CheckedListBox
    $clb.Location = New-Object Drawing.Point(16, 68)
    $clb.Size = New-Object Drawing.Size(612, 185)
    $clb.CheckOnClick = $true
    for ($i = 0; $i -lt $Catalog.Count; $i++) {
        $m = $Catalog[$i]
        $line = "{0,-17} {1}" -f $m.Label, $m.Desc
        [void]$clb.Items.Add($line, $State[$m.Id].Checked)
    }
    $form.Controls.Add($clb)

    $chkCompile = New-Object Windows.Forms.CheckBox
    $chkCompile.Text = "Pre-compile TensorRT engines && calibrate speed presets now (recommended)"
    $chkCompile.Checked = $WantCompile
    $chkCompile.Location = New-Object Drawing.Point(16, 265)
    $chkCompile.Size = New-Object Drawing.Size(612, 24)
    $form.Controls.Add($chkCompile)

    $lblDef = New-Object Windows.Forms.Label
    $lblDef.Text = "Startup model (saved in rife-state.conf; roll anytime with Alt+m / Alt+M):"
    $lblDef.Location = New-Object Drawing.Point(16, 298)
    $lblDef.Size = New-Object Drawing.Size(420, 22)
    $form.Controls.Add($lblDef)

    $cmbDef = New-Object Windows.Forms.ComboBox
    $cmbDef.DropDownStyle = "DropDownList"
    $cmbDef.Location = New-Object Drawing.Point(440, 295)
    $cmbDef.Size = New-Object Drawing.Size(188, 26)
    $form.Controls.Add($cmbDef)

    $refreshCombo = {
        param([string]$preferId)
        $prev = if ($preferId) { $preferId } elseif ($cmbDef.SelectedItem) { $cmbDef.SelectedItem.ToString() } else { "v4.6" }
        $cmbDef.Items.Clear()
        [void]$cmbDef.Items.Add("v4.6")
        for ($i = 0; $i -lt $Catalog.Count; $i++) {
            if ($clb.GetItemChecked($i)) {
                [void]$cmbDef.Items.Add($Catalog[$i].Id)
            }
        }
        $idx = $cmbDef.Items.IndexOf($prev)
        $cmbDef.SelectedIndex = if ($idx -ge 0) { $idx } else { 0 }
    }
    & $refreshCombo $currentDefault

    $clb.Add_ItemCheck({
        param($sender, $e)
        $form.BeginInvoke([Action]{
            # Keep previous selection or switch to newly checked model
            & $refreshCombo ""
        }) | Out-Null
    })

    $btnAll = New-Object Windows.Forms.Button
    $btnAll.Text = "Select All"
    $btnAll.Location = New-Object Drawing.Point(16, 350)
    $btnAll.Size = New-Object Drawing.Size(92, 32)
    $btnAll.Add_Click({
        for ($i = 0; $i -lt $clb.Items.Count; $i++) { $clb.SetItemChecked($i, $true) }
        & $refreshCombo ""
    })
    $form.Controls.Add($btnAll)

    $btnNone = New-Object Windows.Forms.Button
    $btnNone.Text = "Clear All"
    $btnNone.Location = New-Object Drawing.Point(116, 350)
    $btnNone.Size = New-Object Drawing.Size(92, 32)
    $btnNone.Add_Click({
        for ($i = 0; $i -lt $clb.Items.Count; $i++) { $clb.SetItemChecked($i, $false) }
        & $refreshCombo "v4.6"
    })
    $form.Controls.Add($btnNone)

    $btnOk = New-Object Windows.Forms.Button
    $btnOk.Text = "Apply Changes"
    $btnOk.DialogResult = [Windows.Forms.DialogResult]::OK
    $btnOk.Location = New-Object Drawing.Point(432, 350)
    $btnOk.Size = New-Object Drawing.Size(110, 32)
    $form.AcceptButton = $btnOk
    $form.Controls.Add($btnOk)

    $btnCancel = New-Object Windows.Forms.Button
    $btnCancel.Text = "Cancel"
    $btnCancel.DialogResult = [Windows.Forms.DialogResult]::Cancel
    $btnCancel.Location = New-Object Drawing.Point(548, 350)
    $btnCancel.Size = New-Object Drawing.Size(80, 32)
    $form.CancelButton = $btnCancel
    $form.Controls.Add($btnCancel)

    if ($form.ShowDialog() -ne [Windows.Forms.DialogResult]::OK) {
        return $null
    }
    for ($i = 0; $i -lt $Catalog.Count; $i++) {
        $State[$Catalog[$i].Id].Checked = $clb.GetItemChecked($i)
    }
    return [pscustomobject]@{
        Compile = $chkCompile.Checked
        Default = $cmbDef.SelectedItem.ToString()
    }
}

function Show-ConsoleSelector {
    while ($true) {
        Clear-Host
        Write-Host "mpv-NFUA — On-Demand RIFE Model Manager" -ForegroundColor Cyan
        Write-Host "Install folder: $InstallDir`n"
        Write-Host "  [x] RIFE v4.6         Built-in default (always kept)" -ForegroundColor Green
        for ($i = 0; $i -lt $Catalog.Count; $i++) {
            $m = $Catalog[$i]
            $mark = if ($State[$m.Id].Checked) { "x" } else { " " }
            Write-Host ("  [{0}] {1}. {2,-16} {3}" -f $mark, ($i + 1), $m.Label, $m.Desc)
        }
        $ans = Read-Host "`nToggle 1-$($Catalog.Count) (or comma list), A=all, N=none, Enter=apply, Q=quit"
        if (-not $ans) { break }
        $t = $ans.Trim().ToUpperInvariant()
        if ($t -eq "Q") { exit 0 }
        elseif ($t -eq "A") { foreach ($m in $Catalog) { $State[$m.Id].Checked = $true } }
        elseif ($t -eq "N") { foreach ($m in $Catalog) { $State[$m.Id].Checked = $false } }
        else {
            foreach ($tok in ($t -split ",")) {
                $n = 0
                if ([int]::TryParse($tok.Trim(), [ref]$n) -and $n -ge 1 -and $n -le $Catalog.Count) {
                    $id = $Catalog[$n - 1].Id
                    $State[$id].Checked = -not $State[$id].Checked
                }
            }
        }
    }
}

$savedStart = Read-SavedState $StateFile
if (-not $savedStart) { $savedStart = "v4.6" }

if ($Install -or $Remove) {
    if ($Install) {
        if ($Install.Trim().ToLowerInvariant() -eq "all") {
            foreach ($m in $Catalog) { $State[$m.Id].Checked = $true }
        } else {
            foreach ($tok in ($Install -split ",")) {
                if (-not $tok.Trim()) { continue }
                $m = Find-Model $tok
                if (-not $m) { throw "Unknown model ID: $tok" }
                $State[$m.Id].Checked = $true
            }
        }
    }
    if ($Remove) {
        if ($Remove.Trim().ToLowerInvariant() -eq "all") {
            foreach ($m in $Catalog) { $State[$m.Id].Checked = $false }
        } else {
            foreach ($tok in ($Remove -split ",")) {
                if (-not $tok.Trim()) { continue }
                $m = Find-Model $tok
                if (-not $m) { throw "Unknown model ID: $tok" }
                $State[$m.Id].Checked = $false
            }
        }
    }
} elseif (-not $NoPrompt) {
    $guiOk = $false
    try {
        if ([Environment]::UserInteractive) {
            $res = Show-GuiSelector $savedStart
            $guiOk = $true
            if ($null -eq $res) { Write-Host "Cancelled."; exit 0 }
            $WantCompile = $res.Compile
            if (-not $SetDefault) { $SetDefault = $res.Default }
        }
    } catch {
        if ($guiOk) { throw }
    }
    if (-not $guiOk) {
        Show-ConsoleSelector
    }
} elseif (-not $SetDefault) {
    throw "Nothing specified with -NoPrompt (use -Install, -Remove, -SetDefault, or -List)."
}

# 1. Patcher on top: scripts/rife.lua + input.conf + rife.conf
if (-not $NoPatch) {
    Write-Step "Patching player scripts & hotkeys (Alt+m / Alt+M + persistence)"
    New-Item -ItemType Directory -Force -Path $ScriptsDir, $OptsDir | Out-Null

    $batDir = if ($env:MPV_NFUA_BAT_DIR) { $env:MPV_NFUA_BAT_DIR.TrimEnd("\") } else { (Get-Location).Path }
    $srcLua = $null
    $candRoots = @($PatchSource, (Join-Path $batDir "source"), $batDir) | Where-Object { $_ }
    foreach ($root in $candRoots) {
        $cand = Join-Path $root "portable_config\scripts\rife.lua"
        if ((Test-Path $cand) -and ((Get-Content -Raw -LiteralPath $cand) -match "rife-model")) {
            $srcLua = $cand
            break
        }
    }

    $luaTarget = Join-Path $ScriptsDir "rife.lua"
    if ((Test-Path $luaTarget) -and ((Get-Content -Raw -LiteralPath $luaTarget) -match "rife-model")) {
        Write-Host "  scripts\rife.lua already supports multi-model rolling & persistence."
    } else {
        $tmpLua = "$luaTarget.tmp"
        if ($srcLua) {
            Copy-Item -LiteralPath $srcLua -Destination $tmpLua -Force
            Move-Item -LiteralPath $tmpLua -Destination $luaTarget -Force
            Write-Host "  Patched scripts\rife.lua from $srcLua"
        } else {
            try {
                Download-OrCopy "$RepoRawUrl/portable_config/scripts/rife.lua" $tmpLua "rife.lua"
                if ((Get-Content -Raw -LiteralPath $tmpLua) -match "rife-model") {
                    Move-Item -LiteralPath $tmpLua -Destination $luaTarget -Force
                    Write-Host "  Updated scripts\rife.lua from GitHub."
                } else {
                    Remove-Item -Force $tmpLua -ErrorAction SilentlyContinue
                    Write-Warning "Fetched rife.lua did not contain rife-model; skipping."
                }
            } catch {
                Remove-Item -Force $tmpLua -ErrorAction SilentlyContinue
                Write-Warning "Could not download updated scripts\rife.lua ($($_.Exception.Message))."
            }
        }
    }

    if (-not (Test-Path $InputConf)) {
        [IO.File]::WriteAllText($InputConf, "", (New-Object Text.UTF8Encoding($false)))
    }
    $inputTxt = [IO.File]::ReadAllText($InputConf, [Text.Encoding]::UTF8)
    if ($inputTxt -match "script-message rife-model") {
        Write-Host "  input.conf already binds Alt+m / Alt+M."
    } else {
        $block = "# Step through installed RIFE models (Alt+m next, Alt+M previous). Saved across restarts.`r`nAlt+m script-message rife-model cycle`r`nAlt+M script-message rife-model prev`r`n`r`n"
        if ($inputTxt.Contains("# Measure this PC")) {
            $inputTxt = $inputTxt.Replace("# Measure this PC", $block + "# Measure this PC")
        } else {
            $inputTxt = $inputTxt.TrimEnd("`r", "`n") + "`r`n`r`n" + $block
        }
        [IO.File]::WriteAllText($InputConf, $inputTxt, (New-Object Text.UTF8Encoding($false)))
        Write-Host "  Added Alt+m / Alt+M bindings to portable_config\input.conf"
    }

    $rifeConf = Join-Path $OptsDir "rife.conf"
    if (Test-Path $rifeConf) {
        $rcTxt = [IO.File]::ReadAllText($rifeConf, [Text.Encoding]::UTF8)
        if ($rcTxt -notmatch "model=") {
            $add = "# Force a startup RIFE model (empty = remember last model picked with Alt+m / Alt+M).`r`n# Extra models can be added on demand with add-rife-models.sh / Add-RIFE-Models.bat.`r`n#model=v4.6`r`n`r`n"
            if ($rcTxt.Contains("# The screen's refresh rate")) {
                $rcTxt = $rcTxt.Replace("# The screen's refresh rate", $add + "# The screen's refresh rate")
            } else {
                $rcTxt = $add + $rcTxt
            }
            [IO.File]::WriteAllText($rifeConf, $rcTxt, (New-Object Text.UTF8Encoding($false)))
            Write-Host "  Updated portable_config\script-opts\rife.conf comments."
        }
    }
}

# 2. Remove unchecked models + cached engines + presets
$removedAny = @()
foreach ($m in $Catalog) {
    $st = $State[$m.Id]
    if ($st.Installed -and -not $st.Checked) {
        if ($removedAny.Count -eq 0) { Write-Step "Removing unchecked models & freeing engine cache" }
        $onnxPath = Join-Path $RifeDir $m.Onnx
        Remove-Item -LiteralPath $onnxPath -Force -ErrorAction SilentlyContinue
        Get-ChildItem -LiteralPath $RifeDir -Filter ($m.Onnx + ".*.engine") -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath (Join-Path $OptsDir "rife-preset-$($m.Id).conf") -Force -ErrorAction SilentlyContinue
        Write-Host "  Removed $($m.Label) ($($m.Onnx) + cached engines)"
        $st.Installed = $false
        $removedAny += $m.Id
    }
}
if ($removedAny -contains $savedStart) {
    Write-SavedState $StateFile "v4.6"
    Write-Host "  Reset saved startup model from $savedStart back to v4.6."
}

# 3. Install newly checked models with SHA256 verification
New-Item -ItemType Directory -Force -Path $RifeDir | Out-Null
$addedModels = @()
foreach ($m in $Catalog) {
    $st = $State[$m.Id]
    if (-not $st.Installed -and $st.Checked) {
        if ($addedModels.Count -eq 0) { Write-Step "Downloading checked RIFE models" }
        $dst = Join-Path $RifeDir $m.Onnx
        $tmp = "$dst.part"
        $src = if ($BaseUrl -match "^https?://") { $BaseUrl.TrimEnd("/") + "/" + $m.Onnx } else { Join-Path $BaseUrl $m.Onnx }
        Write-Host "  [$($m.Label)] Fetching $src..."
        try {
            Download-OrCopy $src $tmp $m.Onnx
            $hash = (Get-FileHash -LiteralPath $tmp -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($hash -ne $m.Sha256) {
                throw "SHA256 mismatch for $($m.Onnx) (expected $($m.Sha256), got $hash)"
            }
            Move-Item -LiteralPath $tmp -Destination $dst -Force
            Write-Host "  [SHA256 OK] Saved $($m.Onnx) ($($m.Bytes) bytes)" -ForegroundColor Green
            $st.Installed = $true
            $addedModels += $m
        } catch {
            Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
            throw
        }
    }
}

# 4. Pre-compile TensorRT engines & calibrate presets
$benchExe = Join-Path $InstallDir "rife-bench.exe"
if ($WantCompile -and $addedModels.Count -gt 0) {
    if (-not (Test-Path $benchExe)) {
        Write-Warning "rife-bench.exe not found; TensorRT engines will compile on first playback."
    } else {
        Write-Step "Compiling TensorRT engines & calibrating newly added models"
        $screenFps = 160
        try {
            $rates = Get-CimInstance Win32_VideoController | Where-Object { $_.CurrentRefreshRate -ge 20 } | Sort-Object CurrentRefreshRate -Descending | Select-Object -First 1 -ExpandProperty CurrentRefreshRate
            if ($rates) { $screenFps = [int]$rates }
        } catch {}

        foreach ($m in $addedModels) {
            Write-Host "  Compiling & timing $($m.Label) (align=$($m.Align))..."
            $prevErr = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            $out = & $benchExe --model (Join-Path $RifeDir $m.Onnx) --align $m.Align --sizes "1280x720,1920x1080,2560x1440,3840x2160" --runs 10 2>&1
            $rc = $LASTEXITCODE
            $ErrorActionPreference = $prevErr
            $outStr = ($out | Out-String)
            if ($rc -ne 0 -and $outStr -match "unknown option --align") {
                Write-Warning "Installed rife-bench.exe is older and lacks --align; skipping offline calibration (mpv will auto-compile $($m.Label) on first playback)."
                break
            } elseif ($rc -ne 0) {
                Write-Warning "rife-bench.exe failed for $($m.Label) (exit $rc)"
                continue
            }
            $gpu = "NVIDIA GPU"
            $pts = @()
            foreach ($line in $out) {
                $s = $line.ToString().Trim()
                if ($s -match "^gpu\s+(.+)$") { $gpu = $Matches[1].Trim() }
                elseif ($s -match "^result\s+full\s+\S+\s+([0-9.]+)\s+([0-9.]+)$") {
                    Write-Host "    $s"
                    $pts += "$($Matches[1]):$($Matches[2])"
                }
            }
            if ($pts.Count -gt 0) {
                New-Item -ItemType Directory -Force -Path $OptsDir | Out-Null
                $stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
                $presetPath = Join-Path $OptsDir "rife-preset-$($m.Id).conf"
                $presetTxt = @(
                    "# Written by Add-RIFE-Models.bat on $stamp.",
                    "# Model: $($m.Label) ($($m.Onnx))",
                    "# Alt+9 in the player measures again; delete this file to measure at the next start.",
                    "# A setting in rife.conf wins over the same one here.",
                    "",
                    "model=$($m.Id)",
                    "measured=$stamp",
                    "gpu=$gpu",
                    "screen_fps=$screenFps",
                    "",
                    "# GPU time for one interpolated frame, as tensor megapixels:ms (sizes 1280x720,1920x1080,2560x1440,3840x2160).",
                    "full=$($pts -join ',')"
                ) -join "`r`n"
                [IO.File]::WriteAllText($presetPath, $presetTxt + "`r`n", (New-Object Text.UTF8Encoding($false)))
                Write-Host "    Wrote script-opts\rife-preset-$($m.Id).conf"
            }
        }
    }
}

# 5. Save default startup model
if ($SetDefault) {
    if ($SetDefault.Trim().ToLowerInvariant() -eq "v4.6") {
        Write-SavedState $StateFile "v4.6"
        Write-Host "`nStartup model saved: RIFE v4.6 (rife-state.conf)"
    } else {
        $dm = Find-Model $SetDefault
        if (-not $dm) { throw "Unknown model ID for -SetDefault: $SetDefault" }
        if (-not $State[$dm.Id].Installed) { throw "Cannot set $($dm.Id) as default because it is not installed." }
        Write-SavedState $StateFile $dm.Id
        Write-Host "`nStartup model saved: $($dm.Label) (rife-state.conf)" -ForegroundColor Green
    }
}

Write-Step "Done!"
Write-Host "  In mpv-NFUA, press Alt+m (next) or Alt+M (previous) to roll between installed RIFE models."
Write-Host "  Your last chosen model is automatically saved to portable_config\script-opts\rife-state.conf."
