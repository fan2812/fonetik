@echo off
set MYSQL_PATH=C:\xampp\mysql\bin
set BACKUP_DIR=C:\fonetik\backups
set TIMESTAMP=%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%
set TIMESTAMP=%TIMESTAMP: =0%

if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

"%MYSQL_PATH%\mysqldump.exe" -u root penelusuran_fonetik > "%BACKUP_DIR%\backup_%TIMESTAMP%.sql"

forfiles /p "%BACKUP_DIR%" /m *.sql /d -7 /c "cmd /c del @path" 2>nul