# Anaconda packages Configuration

## Entropy

### Requirements

Python: 3.7.2

Package: [entropy](https://pypi.org/project/entropy/) 0.11 

https://github.com/raphaelvallat/entropy
https://raphaelvallat.com/entropy/build/html/index.html
https://pypi.org/project/EntroPy-Package/



Operating System: __Win10__ x64.

### Step by step guide

* Install Anaconda (conda 4.7.10)

* Install [MinGW-w64](https://sourceforge.net/projects/mingw-w64/)

* Create a Python environment with needed dependencies (probably optional, but I always use environments)
```shell
    conda create -c anaconda --name pyenv python=3.7.2 libpython mingw
```

    
* Configure [g++ compiler](https://wiki.python.org/moin/WindowsCompilers)
    * Add MinGW to the path
    ```cmd
        set PATH=%PATH%;C:\Program Files\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin
    ```
    
    * Test if g++ can be executed

    ```cmd
            g++
            g++: fatal error: no input files
            compilation terminated.
            
    ```
    
    - Configure distutils.cfg for pyenv
        - Location:\...\anaconda3\envs\pyenv\Lib\distutils 
        - create distutils.cfg:
        ```cmd
        
        ```
           

7.Activate the python environment
```cmd
    activate pyenv
```
    
8.Install entropy (currently 0.11) with pip
```cmd
    pip install entropy
```