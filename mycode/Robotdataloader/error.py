_mess2code = {
    'NORMAL': 0, #正常
    'NO_DATA': 1, #没有取到数据
    'SHORTER_THAN_TIME_SPAN': 2, #取出数据的长度不够time_span分钟
    'LOW_FREQUENCE': 3, #采集频率没有达到期望频率的90%
    'SYSTEM_ERROR': 4, #其他错误
}
_code2mess = {v: k for k, v in _mess2code.items()}

class DataloaderError(Exception):
    def __init__(self, sample_len, message):
        self.sample_len = sample_len
        self.message = message
        self.err_code = _mess2code[message]