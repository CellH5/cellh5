set PYTHONPATH=C:\Users\sommerc\cellh5\pysrc
set PATH=%PATH%;C:\Python27\Lib\site-packages\numpy

python setup.py py2exe
copy /Y tk85.dll .\dist\
SET mver="1.1"
makensis /Dmver=%mver% build-win64-binary.nsi


