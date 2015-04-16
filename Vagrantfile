Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/trusty64"
  config.vm.provider "virtualbox" do |vb|
    vb.memory="1024"
  end
  config.vm.provision :shell, inline: <<-SHELL
sudo apt-get update
sudo apt-get install -y libopencv-dev libboost-dev libboost-thread-dev libboost-program-options-dev libboost-regex-dev libboost-filesystem-dev libboost-program-options-dev
SHELL
end
