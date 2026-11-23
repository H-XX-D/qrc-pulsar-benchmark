"""Authenticated, expiring, replay-resistant ground authority prototype.

Uses a preprovisioned HMAC key supplied by the caller. Keys are never journaled.
Ground changes mission permission; it cannot bypass onboard protective limits.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


class GroundAuthority:
    def __init__(self,mission,key,*,path=None,daily_limit_m_s=1.):
        if not isinstance(key,bytes) or len(key)<32:raise ValueError('At least256 bits of provisioned key required')
        if not mission or not 0<daily_limit_m_s<=1.:raise ValueError('Invalid launch authority')
        self.mission=mission;self.key=key;self.path=None if path is None else Path(path)
        self.launch_limit=float(daily_limit_m_s)
        self.state=dict(sequence=0,mission_enabled=True,daily_limit_m_s=self.launch_limit)
        if self.path is not None and self.path.exists():
            saved=json.loads(self.path.read_text());body=saved['body']
            if body['mission']!=mission or not hmac.compare_digest(saved['mac'],self.mac(body)):
                raise ValueError('Authority state authentication failed')
            self.state=body['state']
            if (type(self.state['sequence']) is not int or self.state['sequence']<0
                    or type(self.state['mission_enabled']) is not bool
                    or not 0<self.state['daily_limit_m_s']<=self.launch_limit):
                raise ValueError('Invalid saved authority')

    def mac(self,body):return hmac.new(self.key,canonical(body),hashlib.sha256).hexdigest()

    def sign(self,body):return dict(body=body,mac=self.mac(body))

    def receive(self,envelope,now_s):
        try:
            body=envelope['body']
            if not hmac.compare_digest(envelope['mac'],self.mac(body)):return False,'authentication'
            required={'mission','sequence','issued_s','not_before_s','expires_s','action','value'}
            if set(body)!=required or body['mission']!=self.mission:return False,'schema_or_mission'
            sequence=body['sequence']
            if type(sequence) is not int or sequence<=self.state['sequence']:return False,'replay'
            times=[body[k] for k in ('issued_s','not_before_s','expires_s')]+[now_s]
            import math
            if any(type(t) not in (int,float) or not math.isfinite(t) for t in times):return False,'epoch'
            if not 0<=body['issued_s']<=body['not_before_s']<=now_s<=body['expires_s']:return False,'epoch'
            state=dict(self.state);state['sequence']=sequence
            if body['action']=='hold_mission' and body['value'] is None:state['mission_enabled']=False
            elif body['action']=='resume_mission' and body['value'] is None:state['mission_enabled']=True
            elif body['action']=='set_daily_limit' and type(body['value']) in (int,float) and 0<body['value']<=self.launch_limit:
                state['daily_limit_m_s']=float(body['value'])
            else:return False,'authority'
            if self.path is not None:
                self.path.parent.mkdir(parents=True,exist_ok=True)
                record=dict(mission=self.mission,state=state);payload=canonical(dict(body=record,mac=self.mac(record)))
                name=None
                try:
                    with tempfile.NamedTemporaryFile(dir=self.path.parent,delete=False) as out:
                        name=out.name;out.write(payload);out.flush();os.fsync(out.fileno())
                    os.replace(name,self.path)
                finally:
                    if name is not None and os.path.exists(name):os.unlink(name)
            self.state=state;return True,'accepted'
        except (KeyError,TypeError,ValueError,OSError):return False,'invalid_or_storage_failure'
