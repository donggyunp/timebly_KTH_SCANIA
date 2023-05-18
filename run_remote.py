import sys
import os

def main():
    print(os.name)

if __name__ == '__main__':
    remote_user = sys.argv[2]
    remote_password = sys.argv[3]
    code_path = sys.argv[4]
    try:
        if sys.argv[1] == 'deploy':
            import paramiko

            # Connect to remote host
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect('remote_hostname_or_IP', username = remote_user, password = remote_password)

            # Setup sftp connection and transmit this script
            sftp = client.open_sftp()
            sftp.put(__file__, code_path)
            sftp.close()

            # Run the transmitted script remotely without args and show its output.
            # SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
            stdout = client.exec_command('python'+ ' ' + code_path)[1]
            for line in stdout:
                # Process each line in the remote output
                print(line)

            client.close()
            sys.exit(0)
    except IndexError:
        pass

    # No cmd-line args provided, run script normally
    main()