class a:
    def __init__(self):
        self.conf_change = 0
        self.conf_keep = 0
        self.decision_change = False
    def pos_chage_judgement(self,pos0, pos_det):
        if ((abs(pos_det[0]) - abs(pos0[0])) ** 2 + (abs(pos_det[1]) - abs(pos0[1])) ** 2) > (4 ** 2)*2:
            self.conf_change = self.conf_change + 1
            self.decision_change = False
        else:
            self.conf_keep = self.conf_keep + 1
        if self.conf_change >= 3:
            self.decision_change = True
            self.conf_change = 0
        elif self.conf_keep >= 3:
            self.decision_change = False
            self.conf_change = 0
b = a()
pos0_t = [0, 0]
pos_det0 = [5, 5]
b.pos_chage_judgement(pos0_t, pos_det0)
print(b.conf_change)
pos_det1 = [6, 6]
b.pos_chage_judgement(pos0_t, pos_det1)
print(b.conf_change)
pos_det2 = [4, 4]
b.pos_chage_judgement(pos0_t, pos_det2)
print(b.conf_change)
pos_det3 = [4, 4]
b.pos_chage_judgement(pos0_t, pos_det3)
print(b.conf_change)
pos_det4 = [4, 4]
b.pos_chage_judgement(pos0_t, pos_det4)
print(b.conf_change)