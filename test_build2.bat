call mkvirtualenv testenv
call workon testenv
call pip install --upgrade pip
call pip install esinet
call pytest %WORKON_HOME%\testenv\Lib\site-packages\esinet\tests
call deactivate
call rmdir %WORKON_HOME%\testenv /s /q