# ghost_images
Программное обеспечение для формирования фантомных изображений.
Включает в себя модули:
1. `gi.experiment` -- формирование фантомных изображений из экспериментальных данных
2. `gi.emulation` -- эмуляция экспериментов по формированию фантомных изображений, в т.ч. в волоконной фантомной оптике
3. `gi.slm` -- эмуляция фазовой и амплитудной модуляции излучения с помощью SLM и DMD

ПО на уровне интерфейсов совместимо с библиотекой быстрого расчета распространения полей `lightprop2d` и библиотекой расчета мод в многомодовых волокнах `pyMMF`.

`gi.experiment` и `gi.emulation` поддерживают вычисления на CPU с помощью `numpy` и на GPU с помощью `cupy`. 
При больших объёмах данных используется `dask.array` для предотвращения утечек памяти и ускорения вычислений. 

Исследование выполнено за счет гранта Российского научного фонда (проект № 21-12-00155). 
