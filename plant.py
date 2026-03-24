class Plant:
    def __init__(self,k,T,Ts):
        self.k = k
        self.T = T
        self.Ts = Ts
        self.state = 0

    def reset_state(self):
        self.state = 0

    def update(self,u):
        self.state = self.state - self.Ts/self.T*self.state+self.k*self.Ts/self.T*u
        
    def output(self):
        return self.state