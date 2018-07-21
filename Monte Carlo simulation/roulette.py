import random
import math

class Roulette:
	def __init__(self):
		self.choices = list(map(str, range(1, 37)))
		self.choices.append('0')
		self.choices.append('00')
		self.result = None
		self.red = ['1', '3', '5', '7', '9', '12', '14', '16', '18', '19', '21', '23', '25', '27', '30', '32', '34', '36']

	def spin(self):
		self.result = self.choices[random.randint(0, 37)]

	def is_red(self, result):
		if result in self.red:
			return True
		else:
			return False

	def straight_up_bet(self, num, amount):
		self.spin()
		if self.result == num:
			return amount*36
		else:
			return -amount

	def split_bet(self, num1, num2, amount):
		if abs(int(num1)-int(num2)) != 3:
			print("please bet on adjacent numbers!")
			return None
		self.spin()
		if self.result == num1 or self.result == num2:
			return amount*18
		else:
			return -amount

	def street_bet(self, num_list, amount):
		num_list.sort()
		if set(num_list) == set(['0', '1', '2']) or set(num_list) == set(['0', '00', '2']) or set(num_list) == set(['00', '2', '3']):
			self.spin()
			if self.result in num_list:
				return amount*12
			else:
				return -amount
		else:
			if len(num_list) != 3 or (int(num_list[0]) + int(num_list[2])) / 2 != int(num_list[1]):
				print("please bet according to street bet rules!")
				return None
			else:
				self.spin()
				if self.result in num_list:
					return amount*12
				else:
					return -amount

	def corner_bet(self, num_list, amount):
		num_list.sort()
		if (int(num_list[0]) + int(num_list[3])) != (int(num_list[1]) + int(num_list[2])):
			print("please bet a corner of 4!")
			return None
		self.spin()
		if self.result in num_list:
			return amount*9
		else:
			return -amount

	def five_bet(self, amount):
		self.spin()
		if self.result in ['0', '00', '1', '2', '3']:
			return amount*7
		else:
			return -amount

	def line_bet(self, num_list, amount):
		if len(num_list) != 6:
			print("please bet on 6 numbers!")
			return None
		num_list.sort()
		if (int(num_list[0]) + int(num_list[5])) == (int(num_list[1]) + int(num_list[4])) == (int(num_list[2]) + int(num_list[3])):
			self.spin()
			if self.result in num_list:
				return amount*6
			else:
				return -amount
		else:
			print("please bet according to line bet rules!")
			return None

	def column_bet(self, col_num, amount):
		lst = list(range(col_num, 37-(3-col_num), 3))
		for i in range(len(lst)):
			lst[i] = str(lst[i])
		self.spin()
		if self.result in lst:
			return amount*3
		else:
			return -amount

	def dozen_bet(self, dozen_num, amount):
		lst = list(range(12*(dozen_num-1)+1, 12*(dozen_num-1)+1 + 12))
		for i in range(len(lst)):
			lst[i] = str(lst[i])
		self.spin()
		if self.result in lst:
			return amount*3
		else:
			return -amount

	def color_bet(self, color, amount):
		self.spin()
		if self.is_red(self.result) == True:
			if color == 'r':
				return amount*3
			else:
				return -amount
		else:
			if color == 'b':
				return amount*2
			else:
				return -amount

	def odd_even_bet(self, bet, amount):
		self.spin()
		if int(self.result) % 2 == 0:
			if bet == 'e':
				return amount*3
			else:
				return -amount
		else:
			if bet == 'o':
				return amount*2
			else:
				return -amount

	def high_low_bet(self, bet, amount):
		self.spin()
		if bet == 'h':
			if int(self.result) > 18:
				return amount*2
			else:
				return -amount
		else:
			if int(self.result) < 19:
				return amount*2
			else:
				return -amount


