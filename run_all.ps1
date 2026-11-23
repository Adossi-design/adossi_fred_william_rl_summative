# This script runs all 40 experiments for one seed, 4 at a time, and it skips the ones that already have a model file.
param([int]$Seed = 2)

# Make the list of commands to run.
$jobs = @()

foreach ($e in 1..10) {
    if (-not (Test-Path "models\dqn\exp${e}_seed$Seed.pt")) {
        $jobs += "training.dqn_training --exp $e --seed $Seed"
    }
}

foreach ($a in "reinforce", "ppo", "a2c") {
    foreach ($e in 1..10) {
        if (-not (Test-Path "models\pg\${a}_exp${e}_seed$Seed.pt")) {
            $jobs += "training.pg_training --algo $a --exp $e --seed $Seed"
        }
    }
}

# Start the commands, never more than 4 at the same time.
foreach ($j in $jobs) {
    while ((Get-Job -State Running).Count -ge 4) {
        Start-Sleep 20
    }
    Start-Job -ScriptBlock {
        param($cmd, $dir)
        Set-Location $dir
        uv run python -m $cmd.Split(" ")
    } -ArgumentList $j, $PWD | Out-Null
}

# Wait for the last ones and show which finished.
Get-Job | Wait-Job | Out-Null
Get-Job | Receive-Job 2>&1 | Select-String -Pattern "done in|Error|Traceback"
