# Figure 2 source: epoch-level metrics exported from the Forge W&B run
# team-support-vectors-idlf26/iwilds/yoloclip_seed785_1792746715618068563.
param(
    [string]$DataPath = (Join-Path $PSScriptRoot 'yoloclip_training_history.csv'),
    [string]$OutputPath = (Join-Path $PSScriptRoot 'yoloclip_training_curves.png')
)

Set-StrictMode -Version Latest
Add-Type -AssemblyName System.Drawing

$culture = [System.Globalization.CultureInfo]::InvariantCulture
$history = @(Import-Csv -LiteralPath $DataPath | ForEach-Object {
    [pscustomobject]@{
        Epoch = [int]$_.epoch
        Train = [double]::Parse($_.train_ce, $culture)
        IdVal = [double]::Parse($_.id_val_ce, $culture)
    }
})
if ($history.Count -ne 20 -or (@($history.Epoch | Sort-Object) -join ',') -ne ((1..20) -join ',')) {
    throw 'Expected exactly one history row for each epoch from 1 through 20.'
}
$history = @($history | Sort-Object Epoch)
$best = $history | Sort-Object IdVal | Select-Object -First 1
if ($best.Epoch -ne 9) {
    throw "Expected the selected IDVal checkpoint at epoch 9, found epoch $($best.Epoch)."
}

$imageWidth = 1800
$imageHeight = 1000
$plotLeft = 170
$plotTop = 125
$plotWidth = 1550
$plotHeight = 720
$yLow = 1.2
$yHigh = 2.0

function Get-X([double]$epoch) {
    return [single]($plotLeft + ($epoch - 1.0) * $plotWidth / 19.0)
}
function Get-Y([double]$value) {
    return [single]($plotTop + ($yHigh - $value) * $plotHeight / ($yHigh - $yLow))
}

$bitmap = [System.Drawing.Bitmap]::new($imageWidth, $imageHeight)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
try {
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $graphics.Clear([System.Drawing.Color]::White)

    $ink = [System.Drawing.Color]::FromArgb(34, 42, 52)
    $muted = [System.Drawing.Color]::FromArgb(91, 103, 117)
    $blue = [System.Drawing.Color]::FromArgb(35, 91, 151)
    $orange = [System.Drawing.Color]::FromArgb(199, 91, 32)
    $gridPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(222, 227, 232), 2)
    $axisPen = [System.Drawing.Pen]::new($ink, 3)
    $trainPen = [System.Drawing.Pen]::new($blue, 6)
    $valPen = [System.Drawing.Pen]::new($orange, 6)
    $selectionPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(112, 119, 127), 3)
    $selectionPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
    $blueBrush = [System.Drawing.SolidBrush]::new($blue)
    $orangeBrush = [System.Drawing.SolidBrush]::new($orange)
    $inkBrush = [System.Drawing.SolidBrush]::new($ink)
    $mutedBrush = [System.Drawing.SolidBrush]::new($muted)
    $font = [System.Drawing.Font]::new('Arial', 27, [System.Drawing.FontStyle]::Regular)
    $smallFont = [System.Drawing.Font]::new('Arial', 23, [System.Drawing.FontStyle]::Regular)
    $boldFont = [System.Drawing.Font]::new('Arial', 26, [System.Drawing.FontStyle]::Bold)
    $center = [System.Drawing.StringFormat]::new()
    $center.Alignment = [System.Drawing.StringAlignment]::Center
    $right = [System.Drawing.StringFormat]::new()
    $right.Alignment = [System.Drawing.StringAlignment]::Far

    foreach ($tick in 12..20) {
        $value = $tick / 10.0
        $y = Get-Y $value
        $graphics.DrawLine($gridPen, $plotLeft, $y, $plotLeft + $plotWidth, $y)
        $graphics.DrawString($value.ToString('0.0', $culture), $font, $inkBrush,
            [single]($plotLeft - 20), [single]($y - 18), $right)
    }
    $graphics.DrawLine($axisPen, $plotLeft, $plotTop, $plotLeft, $plotTop + $plotHeight)
    $graphics.DrawLine($axisPen, $plotLeft, $plotTop + $plotHeight,
        $plotLeft + $plotWidth, $plotTop + $plotHeight)

    foreach ($tick in @(1, 5, 9, 10, 15, 20)) {
        $x = Get-X $tick
        $graphics.DrawLine($axisPen, $x, $plotTop + $plotHeight,
            $x, $plotTop + $plotHeight + 10)
        $graphics.DrawString("$tick", $font, $inkBrush, $x,
            [single]($plotTop + $plotHeight + 20), $center)
    }

    $selectedX = Get-X $best.Epoch
    $graphics.DrawLine($selectionPen, $selectedX, $plotTop,
        $selectedX, $plotTop + $plotHeight)
    $trainPoints = [System.Drawing.PointF[]]@($history | ForEach-Object {
        [System.Drawing.PointF]::new((Get-X $_.Epoch), (Get-Y $_.Train))
    })
    $valPoints = [System.Drawing.PointF[]]@($history | ForEach-Object {
        [System.Drawing.PointF]::new((Get-X $_.Epoch), (Get-Y $_.IdVal))
    })
    $graphics.DrawLines($trainPen, $trainPoints)
    $graphics.DrawLines($valPen, $valPoints)
    foreach ($point in $trainPoints) {
        $graphics.FillEllipse($blueBrush, $point.X - 5, $point.Y - 5, 10, 10)
    }
    foreach ($point in $valPoints) {
        $graphics.FillEllipse($orangeBrush, $point.X - 6, $point.Y - 6, 12, 12)
    }
    $selectedPoint = $valPoints[$best.Epoch - 1]
    $graphics.FillEllipse([System.Drawing.Brushes]::White,
        $selectedPoint.X - 13, $selectedPoint.Y - 13, 26, 26)
    $graphics.DrawEllipse($valPen, $selectedPoint.X - 13, $selectedPoint.Y - 13, 26, 26)

    $graphics.DrawLine($trainPen, 480, 48, 545, 48)
    $graphics.DrawString('Train cross-entropy', $font, $inkBrush, 560, 28)
    $graphics.DrawLine($valPen, 930, 48, 995, 48)
    $graphics.DrawString('IDVal cross-entropy', $font, $inkBrush, 1010, 28)
    $graphics.DrawString('Cross-entropy', $boldFont, $inkBrush, 5, 81)
    $graphics.DrawString('Epoch', $boldFont, $inkBrush,
        [single]($plotLeft + $plotWidth / 2), 915, $center)
    $graphics.DrawString('Selected checkpoint: epoch 9 (IDVal CE 1.4783)',
        $smallFont, $mutedBrush, 865, 140)
    $graphics.DrawString('Training and familiar-camera validation only; no OOD or test data.',
        $smallFont, $mutedBrush, 170, 958)

    $bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
}
finally {
    $graphics.Dispose()
    $bitmap.Dispose()
}
