import re
from warnings import warn

from cn2an import Cn2An
from cn2an import An2Cn
from cn2an.conf import UNIT_CN2AN


class Transform(object):
    def __init__(self) -> None:
        self.all_num = "零一二三四五六七八九"
        self.all_unit = "".join(list(UNIT_CN2AN.keys()))
        self.cn2an = Cn2An().cn2an
        self.an2cn = An2Cn().an2cn
        self.cn_pattern = f"负?([{self.all_num}{self.all_unit}]+点)?[{self.all_num}{self.all_unit}]+"
        self.smart_cn_pattern = f"-?([0-9]+.)?[0-9]+[{self.all_unit}]+"

    def transform(self, inputs: str, method: str = "cn2an") -> str:
        if method == "cn2an":
            inputs = inputs.replace("廿", "二十").replace("半", "0.5").replace("两", "2")
            # date
            inputs = re.sub(
                fr"((({self.smart_cn_pattern})|({self.cn_pattern}))年)?([{self.all_num}十]+月)?([{self.all_num}十]+日)?",
                lambda x: self.__sub_util(x.group(), "cn2an", "date"), inputs)
            # fraction
            inputs = re.sub(fr"{self.cn_pattern}分之{self.cn_pattern}",
                            lambda x: self.__sub_util(x.group(), "cn2an", "fraction"), inputs)
            # percent
            inputs = re.sub(fr"百分之{self.cn_pattern}",
                            lambda x: self.__sub_util(x.group(), "cn2an", "percent"), inputs)
            # celsius
            inputs = re.sub(fr"{self.cn_pattern}摄氏度",
                            lambda x: self.__sub_util(x.group(), "cn2an", "celsius"), inputs)
            # number
            output = re.sub(self.cn_pattern,
                            lambda x: self.__sub_util(x.group(), "cn2an", "number"), inputs)

        elif method == "an2cn":
            # equation
            inputs = re.sub(r"-?\d+([+\-*/]-?\d+)*=-?\d+",
                            lambda x: self.__sub_util(x.group(), "an2cn", "equation"), inputs)
            # date
            inputs = re.sub(r"(\d{2,4}年)?(\d{1,2}月)?(\d{1,2}日)?",
                            lambda x: self.__sub_util(x.group(), "an2cn", "date"), inputs)
            inputs = re.sub(r'(\d{2,4})(/|-)(\d{1,2})(/|-)(\d{1,2})|(\d{2,4})(/|-)(\d{1,2})|(\d{1,2})(/|-)(\d{1,2})',
                            lambda x: self.__sub_util(x.group(), "an2cn", "date"), inputs)
            # fraction
            inputs = re.sub(r"\d+/\d+",
                            lambda x: self.__sub_util(x.group(), "an2cn", "fraction"), inputs)
            # dollar (支持 $123.45 和 123.45$ 两种格式) - 移到plus_minus_multiply之前
            inputs = re.sub(r"(\$-?(\d+\.)?\d+|-?(\d+\.)?\d+\$)",
                            lambda x: self.__sub_util(x.group(), "an2cn", "dollar"), inputs)
            # plus minus multiply
            inputs = re.sub(r"-?\d+([+\-*]-?\d+)*",
                            lambda x: self.__sub_util(x.group(), "an2cn", "plus_minus_multiply"), inputs)
            # percent
            inputs = re.sub(r"-?(\d+\.)?\d+%",
                            lambda x: self.__sub_util(x.group(), "an2cn", "percent"), inputs)
            # celsius
            inputs = re.sub(r"\d+℃",
                            lambda x: self.__sub_util(x.group(), "an2cn", "celsius"), inputs)
            # number (简化正则，处理剩余的数字)
            output = re.sub(r"-?(\d+\.)?\d+",
                            lambda x: self.__sub_util(x.group(), "an2cn", "number") if not any(symbol in x.string[max(0, x.start()-1):x.end()+3] for symbol in ['$', '刀', '℃', '%']) else x.group(), inputs)
        else:
            raise ValueError(f"error method: {method}, only support 'cn2an' and 'an2cn'!")

        return output

    def __sub_util(self, inputs, method: str = "cn2an", sub_mode: str = "number") -> str:

        def replace_date(match):
            # 获取完整的匹配字符串
            full_match = match.group(0)
            
            # 检查不同的日期格式
            # 格式1: (\d{2,4})/(\d{1,2})/(\d{1,2}) - 年月日格式
            year_month_day_pattern = r'(\d{2,4})(/|-)(\d{1,2})(/|-)(\d{1,2})'
            year_month_day_match = re.match(year_month_day_pattern, full_match)
            # 格式2: (\d{2,4})/(\d{1,2}) - 年月格式
            year_month_pattern = r'(\d{2,4})(/|-)(\d{1,2})'
            year_month_match = re.match(year_month_pattern, full_match)
            
            if year_month_day_match:
                year = year_month_day_match.group(1)
                month = year_month_day_match.group(2)
                day = year_month_day_match.group(3)
                output = f"{year}年{month}月{day}日"

            elif year_month_match:
                year = year_month_match.group(1)
                month = year_month_match.group(2)
                output = f"{year}年{month}月"
            else:
                month_day_pattern = r'(\d{1,2})(/|-)(\d{1,2})'
                month_day_match = re.match(month_day_pattern, full_match)
                if month_day_match:
                    month = month_day_match.group(1)
                    day = month_day_match.group(2)
                    output = f"{month}月{day}日"
                else:
                    output = full_match

            # 如果有年，则年前的数字需要一个个转换成中文
            output = re.sub(r"\d+(?=年)", lambda x: self.an2cn(x.group(), "direct"), output)
            
            # 如果都不匹配，返回原字符串
            return output

        def replace_equation(match):
            # 获取完整的匹配字符串
            full_expr = match.group(0)
            
            # 运算符映射
            operator_map = {
                '+': '加',
                '-': '减', 
                '*': '乘',
                '/': '除'
            }
            
            # 处理连续运算的表达式
            # 先分离等号前后的部分
            if '=' in full_expr:
                left_side, right_side = full_expr.split('=', 1)
                
                # 处理等号左边的表达式
                # 使用正则表达式匹配数字和运算符
                left_pattern = r'(-?\d+)([+\-*/])?'
                left_matches = re.findall(left_pattern, left_side)
                
                new_left = ""
                for i, (num, op) in enumerate(left_matches):
                    new_left += num
                    if op:
                        new_left += operator_map.get(op, op)
                
                # 构建最终结果
                result = new_left + "等于" + right_side
                
                return result
            
            return full_expr

        try:
            if inputs:
                if method == "cn2an":
                    if sub_mode == "date":
                        return re.sub(fr"(({self.smart_cn_pattern})|({self.cn_pattern}))",
                                      lambda x: str(self.cn2an(x.group(), "smart")), inputs)
                    elif sub_mode == "fraction":
                        if inputs[0] != "百":
                            frac_result = re.sub(self.cn_pattern,
                                                 lambda x: str(self.cn2an(x.group(), "smart")), inputs)
                            numerator, denominator = frac_result.split("分之")
                            return f"{denominator}/{numerator}"
                        else:
                            return inputs
                    elif sub_mode == "percent":
                        return re.sub(f"(?<=百分之){self.cn_pattern}",
                                      lambda x: str(self.cn2an(x.group(), "smart")), inputs).replace("百分之", "") + "%"
                    elif sub_mode == "celsius":
                        return re.sub(f"{self.cn_pattern}(?=摄氏度)",
                                      lambda x: str(self.cn2an(x.group(), "smart")), inputs).replace("摄氏度", "℃")
                    elif sub_mode == "number":
                        return str(self.cn2an(inputs, "smart"))
                    else:
                        raise Exception(f"error sub_mode: {sub_mode} !")
                else:
                    if sub_mode == "date":
                        # 直接使用 replace_date 函数处理日期格式
                        return replace_date(type('Match', (), {'group': lambda self, n=0: inputs})())
                    elif sub_mode == "equation":
                        # 使用 replace_equation 函数处理数学表达式
                        return replace_equation(type('Match', (), {'group': lambda self, n=0: inputs})())
                    elif sub_mode == "fraction":
                        frac_result = re.sub(r"\d+", lambda x: self.an2cn(x.group(), "low"), inputs)
                        numerator, denominator = frac_result.split("/")
                        return f"{denominator}分之{numerator}"
                    elif sub_mode == "plus_minus_multiply":
                        frac_result = re.sub(r"\d+",
                            lambda x: self.an2cn(x.group(), "low"), inputs)
                        for c, op in zip(["+", "-", "*"], ["加", "减", "乘"]):
                            frac_result = frac_result.replace(c, op)
                        return frac_result
                    elif sub_mode == "celsius":
                        return self.an2cn(inputs[:-1], "low") + "摄氏度"
                    elif sub_mode == "percent":
                        return "百分之" + self.an2cn(inputs[:-1], "low")
                    elif sub_mode == "dollar":
                        # 处理美元符号在前面或后面的情况
                        if inputs.startswith("$"):
                            # $123.45 格式
                            return self.an2cn(inputs[1:], "low") + "刀"
                        elif inputs.endswith("$"):
                            # 123.45$ 格式
                            return self.an2cn(inputs[:-1], "low") + "刀"
                        else:
                            return inputs
                    elif sub_mode == "number":
                        return self.an2cn(inputs, "low")
                    else:
                        raise Exception(f"error sub_mode: {sub_mode} !")
        except Exception as e:
            warn(str(e))
            return inputs
        
        # 确保所有代码路径都有返回值
        return inputs
