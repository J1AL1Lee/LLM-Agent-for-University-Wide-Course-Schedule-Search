[CmdletBinding()]
param(
    [Parameter()]
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),

    [Parameter()]
    [string[]]$Grades = @("2023", "2024", "2025", "2026")
)

$ErrorActionPreference = "Stop"

function Get-ExcelProvider {
    param([string]$ProbePath)

    foreach ($provider in @("Microsoft.ACE.OLEDB.16.0", "Microsoft.ACE.OLEDB.12.0")) {
        $connection = $null
        try {
            $connectionString = "Provider=$provider;Data Source=$ProbePath;Extended Properties='Excel 8.0;HDR=NO;IMEX=1;READONLY=TRUE'"
            $connection = [System.Data.OleDb.OleDbConnection]::new($connectionString)
            $connection.Open()
            $connection.Close()
            return $provider
        }
        catch {
            if ($null -ne $connection) {
                $connection.Dispose()
            }
        }
    }

    throw "Microsoft ACE OLE DB provider is required to read legacy .xls files."
}

function Read-XlsRows {
    param(
        [string]$Path,
        [string]$Provider
    )

    $connectionString = "Provider=$Provider;Data Source=$Path;Extended Properties='Excel 8.0;HDR=NO;IMEX=1;READONLY=TRUE'"
    $connection = [System.Data.OleDb.OleDbConnection]::new($connectionString)
    try {
        $connection.Open()
        $tables = $connection.GetOleDbSchemaTable([System.Data.OleDb.OleDbSchemaGuid]::Tables, $null)
        $sheetName = @($tables | Where-Object { $_.TABLE_NAME -like '*$' } | Select-Object -First 1 -ExpandProperty TABLE_NAME)[0]
        if ([string]::IsNullOrWhiteSpace($sheetName)) {
            throw "No worksheet found in $Path"
        }

        $escapedSheetName = $sheetName.Replace("]", "]]" )
        $command = $connection.CreateCommand()
        $command.CommandText = "SELECT * FROM [$escapedSheetName]"
        $adapter = [System.Data.OleDb.OleDbDataAdapter]::new($command)
        $table = [System.Data.DataTable]::new()
        [void]$adapter.Fill($table)

        $rows = [System.Collections.Generic.List[object]]::new()
        foreach ($dataRow in $table.Rows) {
            $values = [System.Collections.Generic.List[string]]::new()
            foreach ($value in $dataRow.ItemArray) {
                if ($value -is [DBNull]) {
                    $values.Add("")
                }
                else {
                    $values.Add([string]$value)
                }
            }
            $rows.Add([pscustomobject]@{ Cells = $values.ToArray() })
        }
        return $rows
    }
    finally {
        $connection.Dispose()
    }
}

function Get-RowsHash {
    param([object[]]$Rows)

    $rowSeparator = [char]0x001E
    $cellSeparator = [char]0x001F
    $content = ($Rows | ForEach-Object { [string]::Join($cellSeparator, $_.Cells) }) -join $rowSeparator
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($content)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return [Convert]::ToHexString($sha256.ComputeHash($bytes)).ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
    }
}

function ConvertTo-CsvField {
    param([AllowEmptyString()][string]$Value)
    return '"' + $Value.Replace('"', '""') + '"'
}

function Write-CsvRows {
    param(
        [string]$Path,
        [object[]]$Rows
    )

    $utf8 = [System.Text.UTF8Encoding]::new($false)
    $writer = [System.IO.StreamWriter]::new($Path, $false, $utf8)
    try {
        foreach ($row in $Rows) {
            $fields = foreach ($value in $row.Cells) {
                ConvertTo-CsvField -Value ([string]$value)
            }
            $writer.WriteLine([string]::Join(",", $fields))
        }
    }
    finally {
        $writer.Dispose()
    }
}

$resolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$outputRoot = Join-Path $resolvedProjectRoot "output"
$csvRoot = Join-Path $outputRoot "_schedule_csv"
$gradeRoots = foreach ($grade in $Grades) {
    $gradeRoot = Join-Path $outputRoot $grade
    if (-not (Test-Path -LiteralPath $gradeRoot -PathType Container)) {
        throw "Grade directory not found: $gradeRoot"
    }
    $gradeRoot
}

$allXlsFiles = @($gradeRoots | ForEach-Object { Get-ChildItem -LiteralPath $_ -Filter "*.xls" -File })
if ($allXlsFiles.Count -eq 0) {
    throw "No .xls files found under output/$($Grades -join ',')."
}

$provider = Get-ExcelProvider -ProbePath $allXlsFiles[0].FullName
[System.IO.Directory]::CreateDirectory($csvRoot) | Out-Null

$manifestEntries = [System.Collections.Generic.List[object]]::new()
$summary = [System.Collections.Generic.List[object]]::new()

foreach ($grade in $Grades) {
    $gradeRoot = Join-Path $outputRoot $grade
    $csvGradeRoot = Join-Path $csvRoot $grade
    [System.IO.Directory]::CreateDirectory($csvGradeRoot) | Out-Null

    Get-ChildItem -LiteralPath $csvGradeRoot -Filter "*.csv" -File | Remove-Item -Force

    $seenHashes = @{}
    $written = 0
    $duplicates = 0
    $failed = 0
    $xlsFiles = @(Get-ChildItem -LiteralPath $gradeRoot -Filter "*.xls" -File | Sort-Object Name)

    foreach ($xlsFile in $xlsFiles) {
        try {
            $rows = @(Read-XlsRows -Path $xlsFile.FullName -Provider $provider)
            $contentHash = Get-RowsHash -Rows $rows
            $sourceRelative = [System.IO.Path]::GetRelativePath($resolvedProjectRoot, $xlsFile.FullName).Replace("\", "/")

            if ($seenHashes.ContainsKey($contentHash)) {
                $duplicates++
                $manifestEntries.Add([ordered]@{
                    grade = $grade
                    source_file = $sourceRelative
                    status = "duplicate"
                    duplicate_of = $seenHashes[$contentHash]
                    content_sha256 = $contentHash
                    row_count = $rows.Count
                })
                continue
            }

            $csvName = [System.IO.Path]::ChangeExtension($xlsFile.Name, ".csv")
            $csvPath = Join-Path $csvGradeRoot $csvName
            Write-CsvRows -Path $csvPath -Rows $rows
            $csvRelative = [System.IO.Path]::GetRelativePath($resolvedProjectRoot, $csvPath).Replace("\", "/")
            $seenHashes[$contentHash] = $sourceRelative
            $written++

            $manifestEntries.Add([ordered]@{
                grade = $grade
                source_file = $sourceRelative
                csv_file = $csvRelative
                status = "written"
                content_sha256 = $contentHash
                row_count = $rows.Count
            })
        }
        catch {
            $failed++
            $manifestEntries.Add([ordered]@{
                grade = $grade
                source_file = [System.IO.Path]::GetRelativePath($resolvedProjectRoot, $xlsFile.FullName).Replace("\", "/")
                status = "failed"
                error = $_.Exception.Message
            })
        }
    }

    $summary.Add([ordered]@{
        grade = $grade
        xls_files = $xlsFiles.Count
        csv_files = $written
        duplicate_xls_files = $duplicates
        failed_xls_files = $failed
    })
}

$manifest = [ordered]@{
    generated_at = [DateTimeOffset]::Now.ToString("o")
    provider = $provider
    grades = $Grades
    summary = $summary
    files = $manifestEntries
}
$manifestPath = Join-Path $csvRoot "manifest.json"
$manifestJson = $manifest | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText($manifestPath, $manifestJson, [System.Text.UTF8Encoding]::new($false))

$summary | Format-Table -AutoSize
Write-Output "Manifest: $manifestPath"

if (($summary | Measure-Object -Property failed_xls_files -Sum).Sum -gt 0) {
    throw "One or more .xls files failed to convert. See manifest.json."
}
