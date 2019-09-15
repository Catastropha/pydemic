import torch
from . import factories


def test_memory_set_size():
    am = factories.MemoryFactory.create()

    if not am.memory_size == 1:
        raise AssertionError()
    am.set_size(5)
    if not am.memory_size == 5:
        raise AssertionError()


def test_memory_add():
    am = factories.MemoryFactory.create()

    data = torch.tensor([
        [3.0, 3.0, 3.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])

    am.add(score=1, obj=data[0].clone())
    if not torch.all(torch.eq(data[0], am.memory[0][1])) == True:
        raise AssertionError()
    data[0][0] = 1
    if not torch.all(torch.eq(data[0], am.memory[0][1])) == False:
        raise AssertionError()
    am.add(score=2, obj=data[1].clone())
    am.add(score=0, obj=data[2].clone())
    if not len(am.memory) == 1:
        raise AssertionError()
    if not torch.all(torch.eq(data[2], am.memory[0][1])) == True:
        raise AssertionError()
    am.set_size(3)
    am.add(score=2, obj=data[1].clone())
    am.add(score=1, obj=data[0].clone())
    if not len(am.memory) == 3:
        raise AssertionError()
    if not torch.all(torch.eq(data[2], am.memory[0][1])) == True:
        raise AssertionError()
    if not torch.all(torch.eq(data[0], am.memory[1][1])) == True:
        raise AssertionError()
    if not torch.all(torch.eq(data[1], am.memory[2][1])) == True:
        raise AssertionError()
    data[1][0] = 66
    data[2][0] = 66
    if not torch.all(torch.eq(data[1], am.memory[2][1])) == False:
        raise AssertionError()
    if not torch.all(torch.eq(data[2], am.memory[0][1])) == False:
        raise AssertionError()


def test_memory_topk():
    am = factories.MemoryFactory.create()

    data = torch.tensor([
        [3.0, 3.0, 3.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])

    am.set_size(3)
    am.add(score=0, obj=data[2].clone())
    am.add(score=1, obj=data[0].clone())
    am.add(score=2, obj=data[1].clone())

    result = am.topk(k=2)
    if not len(result) == 2:
        raise AssertionError()
    if not torch.all(torch.eq(data[2], result[0])) == True:
        raise AssertionError()
    if not torch.all(torch.eq(data[0], result[1])) == True:
        raise AssertionError()
    result[0][0] = 66
    if not torch.all(torch.eq(am.memory[0][1], result[0])) == True:
        raise AssertionError()

