"""Calibration text for the importance matrix (imatrix) of small and medium models.

Written for HFQA and dedicated to the public domain (CC0-1.0). It contains no downloaded, user or private material.
It is deliberately varied (English prose and dialogue, German, Chinese, code, data formats, arithmetic) so that the activations it produces are not those of a single
style. It is short compared with what large calibration sets offer (a few thousand tokens); that is stated on the model card and no quality claim is made from it.
"""
from __future__ import annotations

import hashlib

DATASET_ID = "hfqa-original-calibration-v2"
DATASET_LICENSE = "CC0-1.0"

_SECTIONS = [
    """A small library opens beside the river. Each morning a reader returns a book, and the librarian records the title, the date and the shelf number. Clear records help the next visitor find the right volume. The windows face a garden of green trees, and on warm days the doors stay open so that the smell of cut grass drifts between the shelves.

Water can exist as a solid, a liquid or a gas. Heating changes its state while the amount of matter is conserved in a closed container. A careful experiment lists its measurements and units, repeats the observation, and describes the uncertainty. When two results differ, the first question is not who is right but what was different between the two runs.

To prepare a meal, wash the vegetables and cut them into even pieces. Warm a pan, add the ingredients, stir gently and taste before serving. Write down a useful recipe so that another cook can follow the same steps with the same quantities, and note how long each step takes.

The night sky changes with the seasons. Earth rotates on its axis and travels around the Sun. A map can mark the stars that are visible from a chosen place and time. Clouds may hide them, so an observer records the weather along with the hour.""",
    """User: Can you explain what a hash table is?
Assistant: A hash table stores key-value pairs. A hash function turns each key into a number, and that number selects a slot in an array. Looking up a key means computing its hash and reading that slot, so the average cost does not grow with the number of entries. Two keys can land in the same slot; this is called a collision, and implementations resolve it by chaining entries in a list or by probing for the next free slot.
User: Why not keep everything in a sorted list instead?
Assistant: A sorted list gives ordered traversal and fast range queries, and a binary search finds a key in logarithmic time. A hash table is faster for single lookups but keeps no order. The right choice depends on the questions you ask most often.
User: Please summarize that in one sentence.
Assistant: Use a hash table for quick exact lookups and a sorted structure when order or ranges matter.""",
    """Eine Reise beginnt mit einem Plan. Der Zug fährt am Morgen ab und erreicht die Stadt am Mittag. Am Bahnhof stehen Wegweiser, die Reisenden lesen die Anzeige, suchen den richtigen Bahnsteig und lassen anderen Menschen genug Platz. Wer unterwegs eine Pause braucht, findet in der Nähe ein Café mit frischem Brot, Kaffee und einer Karte der Umgebung.

Ein guter Bericht trennt Beobachtungen von Vermutungen. Wenn Belege fehlen, sagt er klar, was unbekannt bleibt. Eine Antwort wird nützlicher, wenn ihre Annahmen und Grenzen sichtbar sind.""",
    """图书馆坐落在河边。每天早晨，读者归还书籍，管理员登记书名、日期和书架编号。清晰的记录能帮助下一位读者找到想要的书。窗外是一片绿树成荫的花园，天气暖和的时候，大门敞开，剪过的青草香气飘进书架之间。

水可以是固态、液态或气态。加热会改变水的状态，而在密闭容器中物质的总量保持不变。严谨的实验会列出测量数据和单位，重复观察，并说明不确定性。当两次结果不同时，首先要问的不是谁对谁错，而是两次实验之间有什么不同。

做饭时，先把蔬菜洗净并切成大小相近的块。预热锅，放入食材，轻轻翻炒，上桌前尝一尝味道。把好用的菜谱写下来，方便别人按相同的步骤和用量重复，并记录每一步需要多长时间。

问：什么是哈希表？
答：哈希表用来存放键值对。哈希函数把每个键变成一个数字，这个数字决定它在数组中的位置。查找时只需计算键的哈希值并读取对应位置，所以平均耗时不会随着元素数量增长。两个键可能落在同一位置，这叫冲突，可以用链表或探测下一个空位来解决。

如果你想把一个复杂的想法讲清楚，可以先用熟悉的词语和一个具体的例子。把观察到的事实和自己的猜测分开写。证据不足的时候，要说明哪些问题仍然未知。""",
    """def total(numbers):
    \"\"\"Return the sum of a list of numbers; an empty list gives zero.\"\"\"
    result = 0
    for value in numbers:
        result += value
    return result


def word_counts(text):
    counts = {}
    for word in text.lower().split():
        word = word.strip(".,;:!?\\"'()")
        if word:
            counts[word] = counts.get(word, 0) + 1
    return counts


class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()


print(total([1, 2, 3]))  # 6""",
    """SELECT customers.name, COUNT(orders.id) AS order_count, SUM(orders.amount) AS revenue
FROM customers
LEFT JOIN orders ON orders.customer_id = customers.id
WHERE orders.created_at >= '2025-01-01'
GROUP BY customers.name
HAVING COUNT(orders.id) > 2
ORDER BY revenue DESC
LIMIT 10;

{"name": "weather-station", "version": "1.4.2", "sensors": [{"id": 1, "type": "temperature", "unit": "C"}, {"id": 2, "type": "humidity", "unit": "%"}], "interval_seconds": 60, "enabled": true}

#!/bin/sh
set -eu
for file in *.log; do
    lines=$(wc -l < "$file")
    echo "$file has $lines lines"
done""",
    """If a train leaves at 9:15 and travels 180 kilometres at 90 kilometres per hour, it arrives at 11:15. A box holds 12 eggs; 7 boxes hold 84 eggs, and after 20 eggs are used, 64 remain. The sum of the first five odd numbers is 1 + 3 + 5 + 7 + 9 = 25, which is 5 squared. Half of 3.5 is 1.75, and 15 percent of 200 is 30. The year 2024 had 366 days because it was a leap year; the following year had 365.

Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday. January, February, March, April, May, June, July, August, September, October, November, December. One, two, three, four, five, six, seven, eight, nine, ten. 零、一、二、三、四、五、六、七、八、九、十。 Eins, zwei, drei, vier, fünf, sechs, sieben, acht, neun, zehn.""",
]

DATASET = ("\n\n".join(_SECTIONS) + "\n").encode("utf-8")
DATASET_SHA256 = hashlib.sha256(DATASET).hexdigest()
