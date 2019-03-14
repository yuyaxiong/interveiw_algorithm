import random
# import pandas as pd

class FSIOHF(object):
    def __init__(self, salary):
        self.salary = salary
        self.tax_rate_old = {1500:0.03, 4500:0.1, 9000:0.2, 35000:0.25, 55000:0.3, 80000:0.35}
        self.tax_rate_new = {3000: 0.03, 12000:0.1, 25000:0.2, 35000:0.25, 55000:0.3, 80000:0.35}
        self.new_base = 5000
        self.old_base = 3000

    def insurance(self):
        yanglao = self.salary * 0.08
        shiye = self.salary * 0.002
        yiliao = self.salary * 0.02
        gongjijing = self.salary * 0.12
        print('INSURANCE:', yanglao, shiye, yiliao, gongjijing, 'ALL:', yanglao + shiye + yiliao + gongjijing)
        return yanglao + shiye + yiliao + gongjijing


    def calc_tax(self, tax_rate, base):
        ins = self.insurance()
        remain = self.salary - ins - base
        tax = 0
        for key in tax_rate:
            if remain <= key:
                tax += tax_rate[key] * remain
                break
            else:
                tax += key * tax_rate[key]
                remain = remain - key
        print('TAX:', tax)
        return self.salary - tax - ins

    def old(self):
        print('OLD:')
        return self.calc_tax(self.tax_rate_old, self.old_base)

    def new(self):
        print('NEW:')
        return self.calc_tax(self.tax_rate_new, self.new_base)

if __name__ == '__main__':
    salary = {9:None, 10:None, 11:None, 12:None, 1:None, 2:None}
    for key in salary:
        s = FSIOHF(18000 + random.randint(0,1) * 200 + 400)
        if key == 9:
            salary[key] = round(s.old(), 1)
        else:
            salary[key] = round(s.new(), 1)
    print(salary)

