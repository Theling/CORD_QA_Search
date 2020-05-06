# To evaluating our project:

* You could open Main.ipynb with Jyputer Notebook, we have seld-explained code and comments. You can see all results displayed in the end of notebook without running our code.

* To run our code, you need to JDK 11 first by following instruction and then run each cell in the notebook.
and download the provided database from https://drive.google.com/file/d/1VRB6xmkUqYqXSSUF2PjeSAXc7vGXZKwb/view?usp=sharing

# Install Dependencies

## Install JDK 11

* Install JDK 11 on Mac
Do followings:

```shell
curl -O https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_osx-x64_bin.tar.gz
tar -xvf openjdk-11.0.2_osx-x64_bin.tar.gz
sudo mv jdk-11.0.2.jdk /Library/Java/JavaVirtualMachines/
```

* Install JDK 11 on Debian Linux (including WSL)

```shell
curl -O https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz
sudo mv openjdk-11.0.2_linux-x64_bin.tar.gz /usr/lib/jvm/; cd /usr/lib/jvm/; tar -zxvf openjdk-11.0.2_linux-x64_bin.tar.gz
sudo update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-11.0.2/bin/java 1
sudo update-alternatives --set java /usr/lib/jvm/jdk-11.0.2/bin/java
```

## Install Python Dependency

This program has been tested on Ubuntu 18.04 and MacOS 10.15. The following instruction are for Ubuntu 18.04.

* Firstly, you need to create a virtual environment using ```conda``` in case messing up your local Python environment

```shell
conda create -n <Env_name> python=3.7
conda activate <Env_name>
```

* Secondly, install packages from provided requirement list ```requirements.txt``` using ```pip``` and ```conda```

```shell
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
```

* If the second step failed, please try to install core packages mannually. This is a list of core packages used in our program:

  * pyserini==0.8.1.0
  * tensorflow==2.0
  * tensorflow_hub
  * pytorch
  * transformers
  * numpy
  * pandas 
  * tqdm
