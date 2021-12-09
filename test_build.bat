call mkvirtualenv testenv
call workon testenv
call pip install C:/Users/Lukas/Documents/projects/esinet
call pytest %WORKON_HOME%\testenv\Lib\site-packages\esinet\tests
call deactivate
call rmdir %WORKON_HOME%\testenv /s /q