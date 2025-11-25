# Doc

Build doc
```shell
make html
```

Auto build
```shell
sphinx-autobuild ./source build/html/
```

If running doc on remote server, using
```shell
sphinx-autobuild ./source build/html/ --host 0.0.0.0 --port 8000
```