# 分類器
class ClassifierBase() :
    def __init__(self) :
        pass
    def train(self) :
        pass
    def test(self) :
        pass
    def train_test(self) :
        pass
    # 特定の引数を抽出
    # @staticmethod
    # def _parse_args(kwargs, group) :
    #     target_keys = [key for key in kwargs.keys() if group in key]
    #     new_kwargs = {key.split('__')[1]:kwargs[key] for key in target_keys}
    #     return new_kwargs