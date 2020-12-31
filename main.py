import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from datetime import datetime as dt


class Digits:
    def __init__(self, digit, start_coord, size, type_digit, help_digit=None, frames_to_change=15):
        """
        :param digit: отображаемая цифра
        :param start_coord: координаты сдвига
        :param size: размер цифры
        :param type_digit: hour0 - первая цифра часа, hour1 - вторая цифра часа
                   minute0 - первая цифра минуты, minute1 - вторая цифра минуты
                   second0 - первая цифра секунды, second1 - вторая цифра секунды
        :param help_digit: используеться для цифры hour1, чтобы реализовать переход между 23 и 00 часов
        :param frames_to_change: количество кадров для анимации перехода
        """
        self.digits = \
            np.array([
                # zero
                [[159.0, 84.0, 123.0, 158.0, 131.0, 258.0],
                 [139.0, 358.0, 167.0, 445.0, 256.0, 446.0],
                 [345.0, 447.0, 369.0, 349.0, 369.0, 275.0],
                 [369.0, 201.0, 365.0, 81.0, 231.0, 75.0]],
                # one
                [[226.0, 99.0, 230.0, 58.0, 243.0, 43.0],
                 [256.0, 28.0, 252.0, 100.0, 253.0, 167.0],
                 [254.0, 234.0, 254.0, 194.0, 255.0, 303.0],
                 [256.0, 412.0, 254.0, 361.0, 255.0, 424.0]],
                # two
                [[152.0, 55.0, 208.0, 26.0, 271.0, 50.0],
                 [334.0, 74.0, 360.0, 159.0, 336.0, 241.0],
                 [312.0, 323.0, 136.0, 454.0, 120.0, 405.0],
                 [104.0, 356.0, 327.0, 393.0, 373.0, 414.0]],
                # three
                [[113.0, 14.0, 267.0, 17.0, 311.0, 107.0],
                 [355.0, 197.0, 190.0, 285.0, 182.0, 250.0],
                 [174.0, 215.0, 396.0, 273.0, 338.0, 388.0],
                 [280.0, 503.0, 110.0, 445.0, 93.0, 391.0]],
                # four
                [[249.0, 230.0, 192.0, 234.0, 131.0, 239.0],
                 [70.0, 244.0, 142.0, 138.0, 192.0, 84.0],
                 [242.0, 30.0, 283.0, -30.0, 260.0, 108.0],
                 [237.0, 246.0, 246.0, 435.0, 247.0, 438.0]],
                # five
                [[226.0, 42.0, 153.0, 44.0, 144.0, 61.0],
                 [135.0, 78.0, 145.0, 203.0, 152.0, 223.0],
                 [159.0, 243.0, 351.0, 165.0, 361.0, 302.0],
                 [371.0, 439.0, 262.0, 452.0, 147.0, 409.0]],
                # six
                [[191.0, 104.0, 160.0, 224.0, 149.0, 296.0],
                 [138.0, 368.0, 163.0, 451.0, 242.0, 458.0],
                 [321.0, 465.0, 367.0, 402.0, 348.0, 321.0],
                 [329.0, 240.0, 220.0, 243.0, 168.0, 285.0]],
                # seven
                [[168.0, 34.0, 245.0, 42.0, 312.0, 38.0],
                 [379.0, 34.0, 305.0, 145.0, 294.0, 166.0],
                 [283.0, 187.0, 243.0, 267.0, 231.0, 295.0],
                 [219.0, 323.0, 200.0, 388.0, 198.0, 452.0]],
                # eight
                [[336.0, 184.0, 353.0, 52.0, 240.0, 43.0],
                 [127.0, 34.0, 143.0, 215.0, 225.0, 247.0],
                 [307.0, 279.0, 403.0, 427.0, 248.0, 432.0],
                 [93.0, 437.0, 124.0, 304.0, 217.0, 255.0]],
                # nine
                [[323.0, 6.0, 171.0, 33.0, 151.0, 85.0],
                 [131.0, 137.0, 161.0, 184.0, 219.0, 190.0],
                 [277.0, 196.0, 346.0, 149.0, 322.0, 122.0],
                 [298.0, 95.0, 297.0, 365.0, 297.0, 448.0]]

            ])
        self.end_points = \
            np.array([
                # zero
                [254, 47],
                # one
                [138, 180],
                # two
                [104, 111],
                # three
                [96, 132],
                # four
                [374, 244],
                # five
                [340, 52],
                # six
                [301, 26],
                # seven
                [108, 52],
                # eight
                [243, 242],
                # nine
                [322, 105]
            ]).astype(float)
        self.digit = digit
        self.shift = start_coord
        self.size = size
        self.img = np.zeros((size[0] + 1, size[1] + 1, 3), dtype=np.uint8)  # массив для отображения цифры
        self.blank = np.zeros((size[0] + 1, size[1] + 1, 3), dtype=np.uint8)  # массив для быстрого обновления self.img
        self.frames_to_change = frames_to_change
        self.type = type_digit
        self.help_digit = help_digit
        self.size_adaptation()

    def size_adaptation(self):
        """
        Данная функция сжимает цифры до переданных размеров
        """
        self.digits[:, :, ::2] = self.digits[:, :, ::2].copy() * self.size[0] / 512
        self.end_points[:, 0] = self.end_points[:, 0].copy() * self.size[0] / 512
        self.digits[:, :, 1::2] = self.digits[:, :, 1::2].copy() * self.size[1] / 512
        self.end_points[:, 1] = self.end_points[:, 1].copy() * self.size[1] / 512
        # self.digits = np.round(self.digits).astype(np.float64)
        # self.end_points = np.round(self.end_points).astype(np.float64)

    def get_coords(self, digit):
        return self.digits[digit]

    def get_shifts(self):
        """
        Данная функция определяет какое максимальное значение может принимать данная цифра часов
        :return: максимальное значение
        """
        if self.type == 'hour0':
            high_digit = 2
        elif self.type == 'minute1' or self.type == 'second1':
            high_digit = 9
        elif self.type == 'minute0' or self.type == 'second0':
            high_digit = 5
        elif self.type == 'hour1':
            if self.help_digit.cur_digit == 2:
                high_digit = 3
            else:
                high_digit = 9
        return high_digit

    def init_change(self, new_digit):
        """
        Данная функция инициальзирует обновление цифры
        :param new_digit: следующая цифра
        :return: первый кадр анимации перехода цифры
        """
        # текущая цифра
        self.cur_digit = self.digit
        # следующая цифра
        self.digit = new_digit
        # текущий фрейм в превращении
        self.cur_step = 0
        # текущие точки
        self.current_besier_points = self.get_coords(self.cur_digit).copy()
        self.current_end_point = self.end_points[self.cur_digit].copy()
        # текущие сдвиги по точкам
        high_digit = self.get_shifts()
        self.current_shift_per_step_vertexes = (self.digits[self.cur_digit + 1 if self.cur_digit < high_digit else 0] -
                                                self.digits[self.cur_digit]) / self.frames_to_change
        self.current_shift_per_step_end_point = (self.end_points[
                                                     self.cur_digit + 1 if self.cur_digit < high_digit else 0] -
                                                 self.end_points[self.cur_digit]) / self.frames_to_change
        return self.plot_digit()

    def casteljau(self, ar_x, ar_y, acc):
        """
        Данная функция инициализирует алгоритм Кастельжо
        :param ar_x: вектор точек x
        :param ar_y: вектор точек y
        :param acc: шаг по t
        :return: набор точек на кривой Безье
        """
        # Запуск рекурсивного алгоритма Кастельжо, acc - шаг по t
        coords_array = []
        for i in range(acc):
            t = float(i) / (acc - 1)  # число для которого вычисляем точку
            x = self.casteljau_rec(ar_x, 0, 3, t).astype(np.int32)
            y = self.casteljau_rec(ar_y, 0, 3, t).astype(np.int32)
            coords_array.append([x, y])  # вычисленные координаты точки
        return coords_array

    def casteljau_rec(self, coord, i, j, t):
        """
        Рекурсивное вычисление точки алгоритмом Кастельжо
        :param coord: вектор точек x или y
        :param i: левая граница контрольных точек
        :param j:  правая граница контрольных точке
        :param t: расстояние от начала отрезка
        :return: сумма кривых меньшего порядка
        """
        # Рекурсивный шаг алгоритма Кастельжо, t - для которого нужно вычислить точку.
        if j == 0:
            return coord[i]
        return self.casteljau_rec(coord, i, j - 1, t) * (1 - t) + self.casteljau_rec(coord, i + 1, j - 1, t) * t

    def draw_line(self, x0, y0, x1, y1):
        """
        Алгоритм Бразенхема
        """
        # x0, y0 = d_vertex[start][0], d_vertex[start][1]
        # x1, y1 = d_vertex[end][0], d_vertex[end][1]
        # вычисляем расстояние
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        # определяем вид линии
        steep = False
        if dy > dx:  # делаем линию более широкой чем длиной
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            dx, dy = dy, dx
            steep = True
        if x0 > x1:  # делаем линию слева направа
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        err = 0
        d_err = dy  # величина ошибки
        y = y0
        # вычисляем сигнум y
        sgn_y = y1 - y0
        sgn_y = 1 if sgn_y > 0 else -1

        # рисуем линию

        for x in range(x0, x1 + 1):
            # рисуем точку
            if steep:
                self.img[x, y] = [255, 255, 255]
            else:
                self.img[y, x] = [255, 255, 255]
            err += d_err  # увеличиваем ошибку
            # если ошибка превысила норму
            if 2 * err >= dx:
                y += sgn_y
                err -= dx

    def __help_plot(self):
        for i in range(4):
            if i == 0:
                coord_x = np.concatenate((np.array([self.current_end_point[0]]), self.current_besier_points[i][::2]))
                coord_y = np.concatenate(
                    (np.array([self.current_end_point[1]]), self.current_besier_points[i][1::2]))
            else:
                coord_x = np.concatenate(
                    (np.array([self.current_besier_points[i - 1][4:][0]]), self.current_besier_points[i][::2]))
                coord_y = np.concatenate(
                    (np.array([self.current_besier_points[i - 1][4:][1]]), self.current_besier_points[i][1::2]))

            coord_array = self.casteljau(coord_x, coord_y, 20)
            for i in range(len(coord_array[:-1])):
                self.draw_line(coord_array[i][0], coord_array[i][1], coord_array[i + 1][0], coord_array[i + 1][1])

    def plot_digit(self):

        # Если нарисовали все фреймы - инициализация новой цифры
        # Иначе - запустить построение кривой безье на основе данных current_besier_points и current_end_point
        # с заданной точностью acc. Связать все полученные точки последовательно через брезенхема и нарисовать.
        # Произвести сдвиг current_besier_points и current_end_point на shift

        self.img = self.blank.copy()
        self.__help_plot()
        self.current_end_point += self.current_shift_per_step_end_point
        self.current_besier_points += self.current_shift_per_step_vertexes
        self.cur_step += 1
        return self.img

    def init_digit(self):
        """
        Данная функция совершает отрисовку цифру при её инициализации
        """
        self.cur_digit = self.digit
        self.current_besier_points = self.get_coords(self.cur_digit).copy()
        self.current_end_point = self.end_points[self.cur_digit].copy()
        self.__help_plot()
        # for i in range(4):
        #     if i == 0:
        #         coordX = np.concatenate((np.array([self.current_end_point[0]]), self.current_besier_points[i][::2]))
        #         coordY = np.concatenate(
        #             (np.array([self.current_end_point[1]]), self.current_besier_points[i][1::2]))
        #     else:
        #         coordX = np.concatenate(
        #             (np.array([self.current_besier_points[i - 1][4:][0]]), self.current_besier_points[i][::2]))
        #         coordY = np.concatenate(
        #             (np.array([self.current_besier_points[i - 1][4:][1]]), self.current_besier_points[i][1::2]))
        #     coordArray = self.casteljau(coordX, coordY, 20)
        #     for j in range(len(coordArray[:-1])):
        #         self.draw_line(coordArray[j][0], coordArray[j][1], coordArray[j + 1][0], coordArray[j + 1][1])
        # print(self.img.tolist())
        # print(self.img.tolist())
        return self.img


class Clock:

    def __init__(self, size=(300, 1350), frames_to_change=15):
        """
        :param size: размер экрана
        :param frames_to_change: количество кадров анимации
        """
        self.frames_to_chang = frames_to_change
        self.finish = False
        self.size = size
        self.old_time = [dt.now().hour, dt.now().minute, dt.now().second]
        # инициализация цифр
        self.hour0 = Digits(0 if len(str(self.old_time[0])) != 2 else self.old_time[0] // 10, (50, 50), (200, 200),
                            'hour0')
        self.hour1 = Digits(self.old_time[0] % 10, (50, 250), (200, 200), 'hour1', self.hour0)
        self.minute0 = Digits(0 if len(str(self.old_time[1])) != 2 else self.old_time[1] // 10, (50, 450), (200, 200),
                              'minute0')
        self.minute1 = Digits(self.old_time[1] % 10, (50, 650), (200, 200), 'minute1')
        self.second0 = Digits(0 if len(str(self.old_time[2])) != 2 else self.old_time[2] // 10, (50, 850), (200, 200),
                              'second0')
        self.second1 = Digits(self.old_time[2] % 10, (50, 1050), (200, 200), 'second1')
        self.digits = [self.hour0, self.hour1, self.minute0, self.minute1, self.second0, self.second1]

        self.changing = False
        self.flags_change = []
        self.fig, self.ax = plt.subplots()
        self.fig.suptitle("Clock")
        self.img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        self.blank = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.im = plt.imshow(self.img, animated=True)
        self.ani = FuncAnimation(self.fig, self.plot_digit, init_func=self.init_clock, frames=self.end_clock, blit=True,
                                 interval=5)
        plt.show()

    def plot_digit(self, par):
        if not self.changing:  # если не происходит анимация перехода
            # обновляем время
            self.new_time = [dt.now().hour, dt.now().minute, dt.now().second]
            self.flags_change = []
            self.new_digits = []
            dif = np.equal(self.new_time, self.old_time)  # проверка на то какие цифры изменились
            # данный цикл заполняет массив об изменение цифр и также заполняет массив с новыми цифрами
            for i in range(len(dif)):
                if not dif[i]:  # если цифра изменилась
                    if len(str(self.new_time[i])) == 2 or self.new_time[i] == 0:
                        # если первая цифра часа или минуты или секунды не равна нулю
                        if self.new_time[i] // 10 != self.old_time[i] // 10:  # если первая цифра изменилась
                            self.flags_change.append(True)
                            self.flags_change.append(True)
                            self.new_digits.append(self.new_time[i] // 10)
                            self.new_digits.append(self.new_time[i] % 10)
                        else:
                            self.flags_change.append(False)
                            self.flags_change.append(True)
                            self.new_digits.append(self.new_time[i] // 10)
                            self.new_digits.append(self.new_time[i] % 10)
                    else:
                        self.flags_change.append(False)
                        self.flags_change.append(True)
                        self.new_digits.append(0)
                        self.new_digits.append(self.new_time[i] % 10)
                    self.changing = True
                else:
                    if len(str(self.new_time[i])) == 2 or self.new_time[i] == 0:
                        self.flags_change.append(False)
                        self.flags_change.append(False)
                        self.new_digits.append(self.new_time[i] // 10)
                        self.new_digits.append(self.new_time[i] % 10)
                    else:
                        self.flags_change.append(False)
                        self.flags_change.append(False)
                        self.new_digits.append(self.new_time[i] % 10)
                        self.new_digits.append(self.new_time[i] % 10)
            # запускаем первый кадр анимации перехода цифр
            for i in range(len(self.digits)):
                if self.flags_change[i]:
                    self.input_img(self.digits[i].shift, self.digits[i].init_change(self.new_digits[i]))
            self.old_time = self.new_time[:]
        else:
            # данный цикл рисует один кадр анимации перехода для каждой изменившейся цифры
            for i in range(len(self.digits)):
                if self.flags_change[i]:
                    self.input_img(self.digits[i].shift, self.digits[i].plot_digit())
                    if self.digits[i].cur_step == self.digits[i].frames_to_change:  # если все цифры поменялись
                        self.changing = False

        self.im.set_array(self.img)
        return self.im,

    def end_clock(self):
        ii = 0
        while not self.finish:
            ii += 1
            yield ii

    def input_img(self, shift, img):
        """
        Данная функция вставляет нужную цифру в часы
        """
        self.img[shift[0]:shift[0] + 200, shift[1]:shift[1] + 200] = img[:-1, :-1].copy()

    def init_clock(self):
        """
        Первичная отрисовка цифр
        :return:
        """
        self.input_img(self.hour0.shift, self.hour0.init_digit())
        self.input_img(self.hour1.shift, self.hour1.init_digit())
        self.input_img(self.minute0.shift, self.minute0.init_digit())
        self.input_img(self.minute1.shift, self.minute1.init_digit())
        self.input_img(self.second0.shift, self.second0.init_digit())
        self.input_img(self.second1.shift, self.second1.init_digit())

        self.im.set_array(self.img)

        return self.im,

    def onclick(self, event):
        self.finish = True


clock = Clock()
# clock.plot_digit(1)


