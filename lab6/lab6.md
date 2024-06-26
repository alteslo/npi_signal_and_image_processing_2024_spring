Для выполнения указанных задач, разработаем экранный интерфейс программы с использованием библиотеки tkinter для графического интерфейса и библиотек opencv, numpy, matplotlib для обработки и визуализации изображений.

Основные шаги:
Наложение шума на изображение:
Реализуем функции добавления гауссовского, "соль и перец" и равномерного шума.
Удаление шумов на изображениях:
Реализуем низкочастотные (усредняющий), высокочастотные (собель/лапласиан) и медианные фильтры.
Повышение контраста изображения:
Реализуем различные виды характеристики передачи уровней: равномерная (линейная), экспоненциальная, Рэлея, степени 2/3, гиперболическая.

Объяснение кода:
Графический интерфейс:

Используется библиотека tkinter для создания оконного интерфейса.
Кнопки для загрузки изображения, добавления шума, удаления шума, повышения контраста и отображения графиков.
Добавление шума:

Гауссовский, "соль и перец" и равномерный шум.
Удаление шума:

Низкочастотный фильтр (усредняющий), высокочастотный фильтр (собель/лапласиан) и медианный фильтр.
Повышение контраста:

Равномерное, экспоненциальное, Рэлея, степени 2/3, гиперболическое изменение контраста.
Отображение изображений:

Используется matplotlib для построения графиков исходного, шумного, фильтрованного изображений и изображения с повышенным контрастом.
Этот интерфейс предоставляет удобные инструменты для обработки изображений и анализа их свойств.