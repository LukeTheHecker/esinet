@REM call pytest -x esinet\tests

call cd tutorials
call pytest -x --nbmake tutorial_1.ipynb
call pytest --nbmake tutorial_2.ipynb
call pytest --nbmake tutorial_3.ipynb
call pytest --nbmake quick_start.ipynb
call pytest --nbmake opm_source.ipynb