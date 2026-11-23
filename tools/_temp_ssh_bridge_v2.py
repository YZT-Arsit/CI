"""Ephemeral authorized SSH bridge. Password is read only from stdin, never saved."""
import sys,json,os,hashlib
import paramiko
HOST="116.172.66.188"
PIN="eb39228d33dd3d3e967d2e6bc07f5c2a6ae37143840177de9de15e804f1b51bc"
print("READY_FOR_EPHEMERAL_PASSWORD",flush=True)
password=sys.stdin.readline().rstrip("\r\n")
c=paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
 c.connect(HOST,port=46481,username="root",password=password,
           allow_agent=False,look_for_keys=False,timeout=20,auth_timeout=20,banner_timeout=25)
finally:
 password=None
actual=hashlib.sha256(c.get_transport().get_remote_server_key().asbytes()).hexdigest()
if actual!=PIN:
 c.close()
 raise RuntimeError("Remote host key pin mismatch; stop, do not execute commands.")
print("SSH_CONNECTED_PIN_VERIFIED",flush=True)
for raw in sys.stdin:
 msg=None
 try:
  msg=json.loads(raw);op=msg.get("op","exec")
  if op=="exit":
   print("BRIDGE_CLOSING",flush=True)
   break
  if op=="exec":
   i,o,e=c.exec_command(msg["cmd"],timeout=240)
   stdout=o.read().decode("utf-8","replace")
   stderr=e.read().decode("utf-8","replace")
   rc=o.channel.recv_exit_status()
   print("RESULT",json.dumps({"id":msg.get("id"),"rc":rc,"stdout":stdout[:36000],"stderr":stderr[:7000]}),flush=True)
  elif op in ("put","get"):
   with c.open_sftp() as ftp:
    if op=="put":
     ftp.put(msg["local"],msg["remote"])
     size=ftp.stat(msg["remote"]).st_size
    else:
     ftp.get(msg["remote"],msg["local"])
     size=os.path.getsize(msg["local"])
   print("RESULT",json.dumps({"id":msg.get("id"),"path":msg["remote"],"bytes":size}),flush=True)
  else:raise ValueError("Invalid op")
 except Exception as err:
  print("ERROR",json.dumps({"id":msg.get("id") if msg else None,"type":type(err).__name__,"message":str(err)[:800]}),flush=True)
c.close()
