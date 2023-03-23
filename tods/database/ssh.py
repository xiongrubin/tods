import paramiko
# 实例化一个transport对象
trans = paramiko.Transport(('192.168.1.112', 22))
# 建立连接
trans.connect(username='root', password='ct@2020')

# 将sshclient的对象的transport指定为以上的trans
ssh = paramiko.SSHClient()
ssh._transport = trans
# 执行命令，和传统方法一样
stdin, stdout, stderr = ssh.exec_command('df -hl')
print(stdout.read().decode())

# 关闭连接
trans.close()

# 建立一个sshclient对象
ssh = paramiko.SSHClient()
# 允许将信任的主机自动加入到host_allow 列表，此方法必须放在connect方法的前面
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# 调用connect方法连接服务器
ssh.connect(hostname='192.168.1.112', port=22, username='root', password='ct@2020')
# 执行命令
stdin, stdout, stderr = ssh.exec_command('ls')
# 结果放到stdout中，如果有错误将放到stderr中
print(stdout.read().decode())
# 关闭连接
ssh.close()
