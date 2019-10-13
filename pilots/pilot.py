'''

pilots.py

Classes representing base pilots.

'''


class BasePilot(object):
    '''
    Base class to define common functions.
    When creating a class, only override the funtions you'd like to replace.
    '''

    def __init__(self, name=None, last_modified=None):
        self.name = name
        self.last_modified = last_modified

    async def decide(self):
        return 0., 0., 0.

    async def start(self):
        pass

    def pname(self):
        return self.name or "Default"
