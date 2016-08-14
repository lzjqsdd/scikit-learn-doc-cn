### 配置过程中出现的问题
####Error1:

	AttributeError: 'module' object has no attribute '_string_to_bool'

####Solution:

	sudo apt-get remove python-matplotlib
  
  参考：[StackOverFlow](http://stackoverflow.com/questions/25383698/error-string-to-bool-in-mplot3d-workaround-found)
  
####error2:

	所有的包最好通过pip install scipy安装，而不是sudo apt-get install python-scipy之类的。
