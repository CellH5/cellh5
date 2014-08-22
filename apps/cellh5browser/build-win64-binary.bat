set PYTHONPATH=C:\Users\sommerc\cellh5\pysrc
set PATH=%PATH%;C:\Python27\Lib\site-packages\numpy

python setup.py py2exe

copy /Y jpeg62.dll .\dist\
copy /Y QtCore4.dll .\dist\
copy /Y QtGui4.dll .\dist\
copy /Y C:\depend64\bin\hdf5_hldll.dll .\dist\
copy /Y C:\depend64\bin\hdf5dll.dll .\dist\

copy /Y C:\depend64\bin\zlib1.dll .\dist\
copy /Y C:\depend64\bin\szip.dll .\dist\

SET mver="1.0"
makensis /Dmver=%mver% build-win64-binary.nsi


