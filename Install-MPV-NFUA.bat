@echo off
setlocal
set "MPV_NFUA_BAT_DIR=%~dp0"
set "PS1=%TEMP%\mpv_nfua_setup_%RANDOM%_%RANDOM%.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$c=[IO.File]::ReadAllText('%~f0',[Text.Encoding]::UTF8); $m='###'+'MPV_NFUA_PS1###'; $i=$c.IndexOf($m); if($i -lt 0){throw 'Payload marker not found.'}; [IO.File]::WriteAllText('%PS1%',$c.Substring($i+$m.Length),(New-Object Text.UTF8Encoding($false)))"
if errorlevel 1 (
    echo Failed to unpack installer payload.
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
set "RC=%ERRORLEVEL%"
del /f /q "%PS1%" >nul 2>&1
echo %* | findstr /i "NoPrompt" >nul
if errorlevel 1 (
    echo.
    if not "%RC%"=="0" echo Setup finished with error code %RC%.
    pause
)
exit /b %RC%
###MPV_NFUA_PS1###
param(
    [string]$InstallDir = "",
    [string]$AppUrl = "https://github.com/NoobFromUA-YT/mpv-NFUA/releases/latest/download/MPV-NFUA-Windows-Portable.zip",
    [switch]$Register,
    [switch]$NoRegister,
    [switch]$NoPrompt,
    [switch]$ForceDownload,
    [switch]$KeepAllSM,
    [switch]$SkipBench
)

$ErrorActionPreference = "Stop"

# Ensure TLS 1.2+ on Windows PowerShell 5.1
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    if ([Enum]::GetNames([Net.SecurityProtocolType]) -contains "Tls13") {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls13
    }
} catch {}

$TrtWheelUrl = "https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-11.4.0.106-py3-none-win_amd64.whl"

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Find-NvidiaSmi {
    $cmd = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $fallback = Join-Path $env:WINDIR "System32\nvidia-smi.exe"
    if (Test-Path $fallback) { return $fallback }
    return $null
}

$batDir = if ($env:MPV_NFUA_BAT_DIR) { $env:MPV_NFUA_BAT_DIR.TrimEnd("\") } else { (Get-Location).Path }

if (-not $InstallDir) {
    if (Test-Path (Join-Path $batDir "mpv.exe")) {
        $InstallDir = $batDir
    } else {
        $defaultDir = Join-Path $env:LOCALAPPDATA "MPV-NFUA"
        if ($NoPrompt) {
            $InstallDir = $defaultDir
        } else {
            Write-Host "MPV-NFUA Windows Installer" -ForegroundColor Green
            $chosen = Read-Host "Install folder [$defaultDir]"
            $InstallDir = if ($chosen -and $chosen.Trim()) { $chosen.Trim().Trim('"') } else { $defaultDir }
        }
    }
}
$InstallDir = [System.IO.Path]::GetFullPath($InstallDir)

Write-Host "MPV-NFUA Windows Setup" -ForegroundColor Green
Write-Host "Install folder: $InstallDir"

Write-Step "Checking NVIDIA GPU & driver"
$nvidiaSmi = Find-NvidiaSmi
if (-not $nvidiaSmi) {
    throw "nvidia-smi.exe not found. An NVIDIA RTX 20 series or newer GPU with a current driver is required."
}

$gpuCsv = & $nvidiaSmi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>$null
if ($LASTEXITCODE -ne 0 -or -not $gpuCsv) {
    throw "Failed to query NVIDIA GPU via nvidia-smi."
}
$firstGpu = ($gpuCsv | Select-Object -First 1).ToString().Split(",")
if ($firstGpu.Count -lt 3) {
    throw "Unexpected nvidia-smi output: $gpuCsv"
}
$gpuName = $firstGpu[0].Trim()
$computeCap = $firstGpu[1].Trim()
$driverVer = $firstGpu[2].Trim()
$smTag = "sm" + ($computeCap -replace "\.", "")

Write-Host "  GPU:         $gpuName"
Write-Host "  Compute Cap: $computeCap ($smTag)"
Write-Host "  Driver:      $driverVer"

if ([int]($computeCap -replace "\.", "") -lt 75) {
    Write-Warning "Compute capability $computeCap is older than Turing (7.5); TensorRT 11.4 may fall back to PTX or fail."
}

if (-not (Test-Path (Join-Path $InstallDir "mpv.exe"))) {
    Write-Step "Downloading MPV-NFUA portable runtime (~85 MB)"
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $zipTmp = Join-Path $env:TEMP ("mpv-nfua-app-" + [Guid]::NewGuid().ToString("N") + ".zip")
    try {
        $req = [System.Net.HttpWebRequest]::Create($AppUrl)
        $req.Method = "GET"
        $req.AllowAutoRedirect = $true
        $req.UserAgent = "MPV-NFUA-Installer/1.0"
        $resp = $req.GetResponse()
        try {
            [long]$totalBytes = $resp.ContentLength
            $stream = $resp.GetResponseStream()
            try {
                $fs = New-Object System.IO.FileStream($zipTmp, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None, 262144)
                try {
                    $buf = New-Object byte[] 131072
                    [long]$done = 0
                    $lastTick = [DateTime]::UtcNow.AddSeconds(-1)
                    while (($n = $stream.Read($buf, 0, $buf.Length)) -gt 0) {
                        $fs.Write($buf, 0, $n)
                        $done += $n
                        $now = [DateTime]::UtcNow
                        if (($now - $lastTick).TotalMilliseconds -ge 500) {
                            if ($totalBytes -gt 0) {
                                $pct = 100.0 * $done / $totalBytes
                                Write-Host ("`r  MPV-NFUA-Windows-Portable.zip  {0,6:F1} / {1,6:F1} MB ({2,5:F1}%)" -f ($done / 1MB), ($totalBytes / 1MB), $pct) -NoNewline
                            } else {
                                Write-Host ("`r  MPV-NFUA-Windows-Portable.zip  {0,6:F1} MB" -f ($done / 1MB)) -NoNewline
                            }
                            $lastTick = $now
                        }
                    }
                    Write-Host ("`r  MPV-NFUA-Windows-Portable.zip  {0,6:F1} MB (100.0%) [OK]        " -f ($done / 1MB))
                } finally { $fs.Dispose() }
            } finally { $stream.Dispose() }
        } finally { $resp.Dispose() }

        Write-Host "  Extracting to $InstallDir..."
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $archive = [System.IO.Compression.ZipFile]::OpenRead($zipTmp)
        try {
            foreach ($entry in $archive.Entries) {
                $rel = $entry.FullName -replace "/", "\"
                if (-not $rel) { continue }
                $dest = [System.IO.Path]::GetFullPath((Join-Path $InstallDir $rel))
                if (-not $dest.StartsWith($InstallDir, [StringComparison]::OrdinalIgnoreCase)) { continue }
                if ($entry.FullName.EndsWith("/") -or $entry.FullName.EndsWith("\")) {
                    New-Item -ItemType Directory -Force -Path $dest | Out-Null
                } else {
                    $parent = [System.IO.Path]::GetDirectoryName($dest)
                    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
                    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
                }
            }
        } finally { $archive.Dispose() }
    } finally {
        Remove-Item $zipTmp -Force -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path (Join-Path $InstallDir "mpv.exe"))) {
        throw "Downloaded archive did not contain mpv.exe."
    }
}

Write-Step "Fetching TensorRT 11.4 DLL index from NVIDIA PyPI wheel"

Add-Type -ReferencedAssemblies @(
    ([System.Net.HttpWebRequest].Assembly.Location),
    ([System.IO.Compression.DeflateStream].Assembly.Location),
    ([System.IO.File].Assembly.Location)
) -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Text;

public class TrtZipEntry {
    public string Name;
    public ushort Method;
    public uint Crc32;
    public long CompSize;
    public long UncompSize;
    public long LocalHeaderOffset;
}

public static class TrtWheelFetcher {
    private static readonly uint[] CrcTable = BuildCrcTable();

    private static uint[] BuildCrcTable() {
        var t = new uint[256];
        for (uint i = 0; i < 256; i++) {
            uint c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }

    private static uint UpdateCrc(uint crc, byte[] buf, int count) {
        crc = ~crc;
        for (int i = 0; i < count; i++)
            crc = CrcTable[(crc ^ buf[i]) & 0xFF] ^ (crc >> 8);
        return ~crc;
    }

    public static string ResolveUrlAndLength(string url, out long length) {
        var req = (HttpWebRequest)WebRequest.Create(url);
        req.Method = "HEAD";
        req.AllowAutoRedirect = true;
        req.UserAgent = "MPV-NFUA-Installer/1.0";
        using (var resp = (HttpWebResponse)req.GetResponse()) {
            length = resp.ContentLength;
            return resp.ResponseUri.AbsoluteUri;
        }
    }

    private static byte[] FetchRange(string url, long start, long end) {
        var req = (HttpWebRequest)WebRequest.Create(url);
        req.Method = "GET";
        req.AllowAutoRedirect = true;
        req.UserAgent = "MPV-NFUA-Installer/1.0";
        req.AddRange(start, end);
        using (var resp = (HttpWebResponse)req.GetResponse())
        using (var s = resp.GetResponseStream())
        using (var ms = new MemoryStream()) {
            byte[] buf = new byte[65536];
            int n;
            while ((n = s.Read(buf, 0, buf.Length)) > 0)
                ms.Write(buf, 0, n);
            return ms.ToArray();
        }
    }

    public static List<TrtZipEntry> ReadCentralDirectory(string resolvedUrl, long fileLength) {
        long tailLen = Math.Min(fileLength, 65536L);
        byte[] tail = FetchRange(resolvedUrl, fileLength - tailLen, fileLength - 1);
        int eocd = -1;
        for (int i = tail.Length - 22; i >= 0; i--) {
            if (tail[i] == 0x50 && tail[i + 1] == 0x4B && tail[i + 2] == 0x05 && tail[i + 3] == 0x06) {
                eocd = i;
                break;
            }
        }
        if (eocd < 0) throw new InvalidDataException("ZIP EOCD not found in remote wheel.");
        uint cdSize = BitConverter.ToUInt32(tail, eocd + 12);
        uint cdOff = BitConverter.ToUInt32(tail, eocd + 16);
        byte[] cd = FetchRange(resolvedUrl, cdOff, cdOff + cdSize - 1);

        var list = new List<TrtZipEntry>();
        int pos = 0;
        while (pos + 46 <= cd.Length) {
            if (cd[pos] != 0x50 || cd[pos + 1] != 0x4B || cd[pos + 2] != 0x01 || cd[pos + 3] != 0x02)
                throw new InvalidDataException("Invalid ZIP central directory signature.");
            ushort method = BitConverter.ToUInt16(cd, pos + 10);
            uint crc = BitConverter.ToUInt32(cd, pos + 16);
            uint comp = BitConverter.ToUInt32(cd, pos + 20);
            uint uncomp = BitConverter.ToUInt32(cd, pos + 24);
            ushort nlen = BitConverter.ToUInt16(cd, pos + 28);
            ushort elen = BitConverter.ToUInt16(cd, pos + 30);
            ushort clen = BitConverter.ToUInt16(cd, pos + 32);
            uint lhoff = BitConverter.ToUInt32(cd, pos + 42);
            string name = Encoding.UTF8.GetString(cd, pos + 46, nlen);
            list.Add(new TrtZipEntry {
                Name = name,
                Method = method,
                Crc32 = crc,
                CompSize = comp,
                UncompSize = uncomp,
                LocalHeaderOffset = lhoff
            });
            pos += 46 + nlen + elen + clen;
        }
        return list;
    }

    public static void ExtractEntry(string resolvedUrl, TrtZipEntry entry, string destPath) {
        byte[] lh = FetchRange(resolvedUrl, entry.LocalHeaderOffset, entry.LocalHeaderOffset + 29);
        if (lh.Length < 30 || lh[0] != 0x50 || lh[1] != 0x4B || lh[2] != 0x03 || lh[3] != 0x04)
            throw new InvalidDataException("Invalid local file header for " + entry.Name);
        ushort lfNlen = BitConverter.ToUInt16(lh, 26);
        ushort lfElen = BitConverter.ToUInt16(lh, 28);
        long dataStart = entry.LocalHeaderOffset + 30 + lfNlen + lfElen;
        long dataEnd = dataStart + entry.CompSize - 1;

        string tmpPath = destPath + ".tmp";
        if (File.Exists(tmpPath)) File.Delete(tmpPath);

        string fileName = Path.GetFileName(destPath);
        double totalMb = entry.UncompSize / 1048576.0;

        try {
            var req = (HttpWebRequest)WebRequest.Create(resolvedUrl);
            req.Method = "GET";
            req.AllowAutoRedirect = true;
            req.UserAgent = "MPV-NFUA-Installer/1.0";
            req.AddRange(dataStart, dataEnd);

            uint crc = 0;
            long written = 0;
            DateTime lastPrint = DateTime.UtcNow.AddSeconds(-1);

            using (var resp = (HttpWebResponse)req.GetResponse())
            using (var rawStream = resp.GetResponseStream()) {
                if (entry.Method != 8 && entry.Method != 0)
                    throw new NotSupportedException("Unsupported ZIP method " + entry.Method);
                Stream srcStream = entry.Method == 8
                    ? (Stream)new DeflateStream(rawStream, CompressionMode.Decompress)
                    : rawStream;

                using (srcStream)
                using (var fs = new FileStream(tmpPath, FileMode.Create, FileAccess.Write, FileShare.None, 262144)) {
                    byte[] buf = new byte[131072];
                    int n;
                    while ((n = srcStream.Read(buf, 0, buf.Length)) > 0) {
                        fs.Write(buf, 0, n);
                        crc = UpdateCrc(crc, buf, n);
                        written += n;
                        DateTime now = DateTime.UtcNow;
                        if ((now - lastPrint).TotalMilliseconds >= 500) {
                            double curMb = written / 1048576.0;
                            double pct = entry.UncompSize > 0 ? (100.0 * written / entry.UncompSize) : 100.0;
                            Console.Write("\r  {0,-42} {1,6:F1} / {2,6:F1} MB ({3,5:F1}%)", fileName, curMb, totalMb, pct);
                            lastPrint = now;
                        }
                    }
                }
            }

            Console.WriteLine("\r  {0,-42} {1,6:F1} / {2,6:F1} MB (100.0%) [CRC OK]", fileName, totalMb, totalMb);

            if (written != entry.UncompSize)
                throw new InvalidDataException("Size mismatch for " + fileName + ": got " + written + ", expected " + entry.UncompSize);
            if (crc != entry.Crc32)
                throw new InvalidDataException(string.Format("CRC32 mismatch for {0}: got {1:x8}, expected {2:x8}", fileName, crc, entry.Crc32));

            if (File.Exists(destPath)) File.Delete(destPath);
            File.Move(tmpPath, destPath);
        } catch {
            if (File.Exists(tmpPath)) File.Delete(tmpPath);
            throw;
        }
    }
}
"@

[long]$wheelLength = 0
$resolvedUrl = [TrtWheelFetcher]::ResolveUrlAndLength($TrtWheelUrl, [ref]$wheelLength)
$entries = [TrtWheelFetcher]::ReadCentralDirectory($resolvedUrl, $wheelLength)

$entryMap = @{}
foreach ($e in $entries) { $entryMap[$e.Name] = $e }

$smDll = "tensorrt_libs/nvinfer_builder_resource_${smTag}_11.dll"
$ptxDll = "tensorrt_libs/nvinfer_builder_resource_ptx_11.dll"

$wanted = New-Object System.Collections.Generic.List[string]
$wanted.Add("tensorrt_libs/nvinfer_11.dll")
$wanted.Add("tensorrt_libs/nvonnxparser_11.dll")
if ($entryMap.ContainsKey($smDll)) {
    $wanted.Add($smDll)
} else {
    Write-Host "  Note: No dedicated $smTag resource DLL in wheel; using PTX fallback." -ForegroundColor Yellow
    $wanted.Add($ptxDll)
}
if ($KeepAllSM -and -not $wanted.Contains($ptxDll)) { $wanted.Add($ptxDll) }

[long]$totalComp = 0
[long]$totalRaw = 0
foreach ($w in $wanted) {
    if (-not $entryMap.ContainsKey($w)) { throw "Expected wheel entry '$w' not found." }
    $totalComp += $entryMap[$w].CompSize
    $totalRaw += $entryMap[$w].UncompSize
}
Write-Host ("  Selected {0} TensorRT DLLs for {1} ({2:N1} MB download -> {3:N1} MB extracted; skips ~1.6 GB of unused GPU architectures)" -f $wanted.Count, $smTag, ($totalComp / 1MB), ($totalRaw / 1MB))

Write-Step "Downloading & verifying TensorRT 11.4 DLLs"
foreach ($w in $wanted) {
    $e = $entryMap[$w]
    $dllName = [System.IO.Path]::GetFileName($w)
    $destPath = Join-Path $InstallDir $dllName
    if (-not $ForceDownload -and (Test-Path $destPath) -and ((Get-Item $destPath).Length -eq $e.UncompSize)) {
        Write-Host ("  {0,-42} already present ({1:F1} MB), skipping" -f $dllName, ($e.UncompSize / 1MB))
        continue
    }
    $attempt = 0
    while ($true) {
        $attempt++
        try {
            [TrtWheelFetcher]::ExtractEntry($resolvedUrl, $e, $destPath)
            break
        } catch {
            if ($attempt -ge 3) { throw }
            Write-Host ""
            Write-Warning "Download attempt $attempt failed for ${dllName}: $($_.Exception.Message). Retrying..."
            Start-Sleep -Seconds 2
        }
    }
}

if (-not $SkipBench) {
    Write-Step "Compiling TensorRT engines & benchmarking this PC (takes ~1-2 minutes once)"
    $benchExe = Join-Path $InstallDir "rife-bench.exe"
    $onnxFull = Join-Path $InstallDir "portable_config\rife\rife_v4.6.onnx"
    $onnxHalf = Join-Path $InstallDir "portable_config\rife\rife_v4.6_half.onnx"
    if (-not (Test-Path $benchExe)) { throw "rife-bench.exe not found in $InstallDir" }

    $prevErrPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $benchOut = @()
    & $benchExe --model $onnxFull --model-half $onnxHalf --runs 20 -v 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Write-Host "  $line"
        $benchOut += $line
    }
    $benchRc = $LASTEXITCODE
    $ErrorActionPreference = $prevErrPref

    if ($benchRc -ne 0) {
        throw "rife-bench.exe failed with exit code $benchRc."
    }

    $benchGpu = $gpuName
    $fullPts = @()
    $halfPts = @()
    foreach ($line in $benchOut) {
        if ($line -match '^gpu\s+(.+?)\s*$') {
            $benchGpu = $Matches[1].Trim()
        } elseif ($line -match '^result\s+(full|half)\s+(\S+)\s+([\d\.]+)\s+([\d\.]+)') {
            $pt = "$($Matches[3]):$($Matches[4])"
            if ($Matches[1] -eq 'full') { $fullPts += $pt } else { $halfPts += $pt }
        }
    }
    if ($halfPts.Count -eq 0) {
        throw "rife-bench.exe produced no benchmark results."
    }

    $screenFps = 160
    try {
        $detected = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
            Where-Object { $_.CurrentRefreshRate -ge 20 } |
            Sort-Object CurrentRefreshRate -Descending |
            Select-Object -First 1 -ExpandProperty CurrentRefreshRate
        if ($detected -and [int]$detected -ge 20) {
            $screenFps = [int]$detected
        }
    } catch {}

    $presetDir = Join-Path $InstallDir "portable_config\script-opts"
    New-Item -ItemType Directory -Force -Path $presetDir | Out-Null
    $presetPath = Join-Path $presetDir "rife-preset.conf"
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    $presetLines = @(
        "# Written by MPV-NFUA Windows setup on $stamp.",
        "# Alt+9 in the player measures again; delete this file to measure at the next start.",
        "# A setting in rife.conf wins over the same one here.",
        "",
        "measured=$stamp",
        "gpu=$benchGpu",
        "screen_fps=$screenFps",
        "",
        "# GPU time for one interpolated frame, as tensor megapixels:ms, at full and",
        "# half flow resolution (sizes 1280x720,1920x1080,2560x1440,3840x2160).",
        ("full=" + ($fullPts -join ",")),
        ("half=" + ($halfPts -join ","))
    )
    [System.IO.File]::WriteAllLines($presetPath, $presetLines, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "  Wrote $presetPath (screen_fps=$screenFps Hz)"
}

$doRegister = $false
if ($Register) {
    $doRegister = $true
} elseif (-not $NoRegister -and -not $NoPrompt) {
    $ans = Read-Host "`nCreate Desktop shortcut and set MPV-NFUA as default mpv? [y/N]"
    if ($ans -match '^[Yy]') { $doRegister = $true }
}

if ($doRegister) {
    Write-Step "Creating Desktop shortcut & registering default mpv"
    $mpvExe = Join-Path $InstallDir "mpv.exe"
    $desktop = [Environment]::GetFolderPath("Desktop")
    $lnkPath = Join-Path $desktop "MPV-NFUA.lnk"
    $ws = New-Object -ComObject WScript.Shell
    $sc = $ws.CreateShortcut($lnkPath)
    $sc.TargetPath = $mpvExe
    $sc.Arguments = "--player-operation-mode=pseudo-gui"
    $sc.WorkingDirectory = $InstallDir
    $sc.IconLocation = "$mpvExe,0"
    $sc.Save()
    Write-Host "  Shortcut: $lnkPath"

    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    $roots = @("HKCU:\Software")
    if ($isAdmin) { $roots += "HKLM:\Software" }

    foreach ($root in $roots) {
        $cmdKey = "$root\Classes\Applications\mpv.exe\shell\open\command"
        $iconKey = "$root\Classes\Applications\mpv.exe\DefaultIcon"
        $appKey = "$root\Microsoft\Windows\CurrentVersion\App Paths\mpv.exe"
        foreach ($k in @($cmdKey, $iconKey, $appKey)) {
            if (-not (Test-Path $k)) { New-Item -Path $k -Force | Out-Null }
        }
        Set-ItemProperty -Path $cmdKey -Name "(default)" -Value "`"$mpvExe`" `"%1`""
        Set-ItemProperty -Path $iconKey -Name "(default)" -Value "`"$mpvExe`",0"
        Set-ItemProperty -Path $appKey -Name "(default)" -Value $mpvExe
        Set-ItemProperty -Path $appKey -Name "Path" -Value $InstallDir
    }

    $scope = if ($isAdmin) { "Machine" } else { "User" }
    $curPath = [Environment]::GetEnvironmentVariable("Path", $scope)
    $normInstall = $InstallDir.TrimEnd("\").ToLowerInvariant()
    $parts = @($curPath -split ";" | Where-Object { $_ -ne "" -and $_.TrimEnd("\").ToLowerInvariant() -ne $normInstall })
    $newParts = @()
    $inserted = $false
    foreach ($p in $parts) {
        if (-not $inserted -and $p.TrimEnd("\").ToLowerInvariant() -eq "c:\mpv") {
            $newParts += $InstallDir
            $inserted = $true
        }
        $newParts += $p
    }
    if (-not $inserted) { $newParts = @($InstallDir) + $newParts }
    [Environment]::SetEnvironmentVariable("Path", ($newParts -join ";"), $scope)

    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ShellNotify {
    [DllImport("shell32.dll")]
    public static extern void SHChangeNotify(int wEventId, int uFlags, IntPtr dwItem1, IntPtr dwItem2);
}
"@
    [ShellNotify]::SHChangeNotify(0x08000000, 0x0000, [IntPtr]::Zero, [IntPtr]::Zero)
    Write-Host "  Registered Applications\mpv.exe, App Paths\mpv.exe, and $scope PATH."
}

$installedBytes = (Get-ChildItem $InstallDir -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ""
Write-Host ("Setup complete! Ready to play ({0:N1} MB total on disk)." -f ($installedBytes / 1MB)) -ForegroundColor Green
Write-Host "Launch from $InstallDir\MPV-Antigravity.bat or drag any video onto mpv.exe."
