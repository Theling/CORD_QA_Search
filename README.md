# To evaluating our project:
* You could open Main.ipynb with Jyputer Notebook, we have seld-explained code and comments. You can see all results displayed in the end of notebook without running our code.

* To run our code, you need to JDK 11 first by following instruction and then run each cell in the notebook.

# Install Dependencies
## Install JDK 11
* Install JDK 11 on Mac
Do followings:
```shell
curl -O https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_osx-x64_bin.tar.gz
tar -xvf openjdk-11.0.2_osx-x64_bin.tar.gz
sudo mv jdk-11.0.2.jdk /Library/Java/JavaVirtualMachines/
```
* Install JDK 11 on Debian (WSL)
```shell
curl -O https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz
sudo mv openjdk-11.0.2_linux-x64_bin.tar.gz /usr/lib/jvm/; cd /usr/lib/jvm/; tar -zxvf openjdk-11.0.2_linux-x64_bin.tar.gz
sudo update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-11.0.2/bin/java 1
sudo update-alternatives --set java /usr/lib/jvm/jdk-11.0.2/bin/java
```
## Install Pyserini
```shell
pip install pyserini==0.8.1.0
```
