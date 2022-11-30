# Genetic Algorithm factor mining with unit check

本项目实现了一个基于遗传算法的自动因子挖掘框架，并附带量纲检查等功能

[This project](https://github.com/yiluuu1/Other_factor_mining) implements an automatic factor mining framework based on genetic algorithm with unit checking etc.

## Table of Contents

- [Background](#background)

This repository contains:

1. [_my_program.py](#Program): The underlying data structure representative for factor.
2. [my_fitness.py](#Fitness): Metrics to evaluate the fitness of a program.
3. [my_funtions.py](#Funtion): The functions used to create programs.
4. [my_genetic.py](#Algorithm): API for Genetic Algorithm factor mining
5. [my_utils.py](#Others): Utilities that are imported by other modules.
6. [split_layer.py](#Factor test): The modules for split layer test for single factor.
7. sample_main.py: The sample test for the whole framework.
8. sample_data: the data used in sample_main.
9. 汇报.pptx: the slides for presetation.

## Background

遗传算法是一种启发式的公式进化技术，通过模拟自然界中遗传进化的过程来逐渐生成契合特定目标的公式群体，适合进行特征工程。
将遗传规划运用于选股因子挖掘时，可以充分利用计算机的强大算力，同时突破人类的思维局限，挖掘出某些隐藏的、难以通过人脑构建的因子。

通过SymbolicRegressor进行回归，SymbolicClassifier进行两分类，以及SymbolicTransformer进行自动化特征工程的转换，
SymbolicTransformer被设计为支持回归问题，但也应该适用于两分类。 *To do: SymbolicClassifier与SymbolicTransformer*

因子进化通过符号回归量是一种估计量，它首先建立一个朴素随机公式的种群来表示一种关系。公式以树状结构表示，数学函数递归地应用于变量和常数。
然后，从种群中选择最适合的个体进行杂交、突变或繁殖等遗传操作，从而从上一代进化出每一代连续的程序。

## Program

Member function `build_program` build an initial random program.
And this class get several evolve operator: `crossover`, `subtree_mutation`, `hoist_mutation`, `point_mutation`,
`crossover` and `reproduce` in the member func. Also, `execute` and `fitness` return the y_hat and fitness.
Additionally, `unit_rationality` return a bool value to tell us if this factor's unit is rationality.

And u can define custom rationality in

```
        rationality = ['weight', 'money', 'time', 'area', 'volume', 'price', 'money/weight', 'weight/time',
                       'weight/area', 'money/volume', 'volume/time', 'volume/area', None]
```

## Fitness

`make_fitness` is a factory function creates a fitness measure object which
measures the quality of a program's fit and thus its likelihood to undergo genetic operations into the next generation.
And we cdefine several `fitness` like `ic`, `ir`, `sharpe` and `mean_square_error `, etc.

### Funtion

It's similar to fitness, `make_function` is a factory function creates a function object.

## Algorithm

`BaseSymbolic.fit` will call function `_evolve` which will execute each generation's evolve.

And function `_tournament` will choose the parent for different evolve operator as following:

```    
    def _tournament():
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders]
        if metric.greater_is_better:
            parents_index = contenders[np.argmax(fitness)]
        else:
            parents_index = contenders[np.argmin(fitness)]
        return parents[parents_index], parents_index
```

## Factor test

`evaluate` Do a 5-layer split test, and check its path dependence.

## Others

`unit_transform` will do a preprocess to unit_dict, transfer it into specific category.
`check_unit` define the different function's unit results.
`preprocess` will do a winsorize, normalize and fillna to final factor.
