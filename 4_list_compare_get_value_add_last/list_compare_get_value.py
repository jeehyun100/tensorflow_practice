


def get_combine_label_list_fn(origin_list, compare_list):

    """ 리스트 두개를 비교하여 차이나는 값만 마지막에 순서대로 넣는 함수
        The function that compare two list and insert distingush values

    Args:
      params:
        * origin_list : A original list
        * compare_list: An compare lists

    Returns:
      list
      두 리스트를 비교하여 구별된 값을 새 원래 리스트에 추가하여 반환
    Raises:
    Example
        origin_list = ['A','B','C','D']
        compare_list = ['50>=', '50=','C','D','E']
        result => ['A', 'B', 'C', 'D', '50=', '50>=', 'E']
    """

    _origin_list = list(origin_list)
    _compare_list = list(compare_list)

    _union_values = set(_origin_list).union(set(_compare_list))
    _diff_values = sorted(list(_union_values - set(_origin_list)))
    _origin_list.extend(_diff_values)

    return _origin_list

if __name__ == "__main__":
    origin_list = ['A','B','C','D']
    #compare_list = ['A','C','E','G','B','Z','S']
    #compare_list = ['A','1','S']
    compare_list = ['50>=', '50=','C','D','E']
    new_label = get_combine_label_list_fn(origin_list, compare_list)
    print(str(new_label))