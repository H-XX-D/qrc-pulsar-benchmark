"""Local append-only telemetry outbox; transport/acknowledgment is caller-owned.

Used with a simulated ground receiver in the fault campaign. This does not send
anything to an external ground station or provide a flight communications stack.
"""
import json
import os
from pathlib import Path


class EventJournal:
    def __init__(self,path=None,*,capacity=4096):
        if type(capacity) is not int or capacity<1:raise ValueError('Positive journal capacity required')
        self.path=None if path is None else Path(path)
        self.capacity=capacity;self.sequence=0;self.pending={};self.maximum_pending=0
        if self.path is not None:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            if self.path.exists():
                for line in self.path.read_text().splitlines():self._apply(json.loads(line))
        if len(self.pending)>capacity:raise OverflowError('Existing outbox exceeds capacity')

    def _apply(self,record):
        if record['record']=='event':
            if record['sequence']!=self.sequence+1:raise ValueError('Journal sequence discontinuity')
            self.sequence=record['sequence'];self.pending[self.sequence]=record
            self.maximum_pending=max(self.maximum_pending,len(self.pending))
        elif record['record']=='ack':
            if record['sequence'] not in self.pending:raise ValueError('Unknown acknowledgment')
            del self.pending[record['sequence']]
        else:raise ValueError('Unknown journal record')

    def _append(self,record):
        payload=json.dumps(record,allow_nan=False,separators=(',',':'))+'\n'
        if self.path is not None:
            with self.path.open('a') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
        self._apply(json.loads(payload))

    def publish(self,time_s,event,details):
        if len(self.pending)>=self.capacity:raise OverflowError('Telemetry outbox capacity exceeded')
        record=dict(record='event',sequence=self.sequence+1,time_s=float(time_s),event=event,details=details)
        self._append(record);return json.loads(json.dumps(record))

    def acknowledge(self,sequence):
        if type(sequence) is not int or sequence not in self.pending:raise ValueError('Unknown acknowledgment')
        self._append(dict(record='ack',sequence=sequence))

    def deliver(self,receiver):
        """Receiver must deduplicate sequence IDs and return True only on receipt."""
        for sequence,record in list(self.pending.items()):
            if receiver(json.loads(json.dumps(record))):self.acknowledge(sequence)
