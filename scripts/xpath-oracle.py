#!/usr/bin/env python3
"""Checks jevons' XPath subset against a reference XPath 1.0 engine (lxml).

The recorded interface's first window becomes an XML document whose elements are named by role
and whose attributes are the element properties. Every expression in the corpus is evaluated
by lxml and by `jevons-desktop --xpath` over the same tree, and their results must agree:
- the same elements (by role and name), in the same order;
- the same attribute values;
- the same numbers, booleans and strings.

Element text is left out: jevons gives an element's text together with its descendants', while
XML text nodes do not exist here.

    pip install lxml
    python3 scripts/xpath-oracle.py [--bin target/debug/jevons-desktop] [--tree examples/desktop/trees/slack.json]
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys

from lxml import etree

CORPUS = [
    # Child and descendant steps, positions among siblings and over a whole result.
    "//TreeItem",
    "//Tree/TreeItem",
    "//Tree//TreeItem[not(@automation_id)]/@name",
    "//TreeItem[1]",
    "//TreeItem[last()]",
    "(//TreeItem)[2]",
    "(//ListItem)[position() > last() - 2]",
    "//Group[@class]/Text[1]",
    "//TreeItem[position() mod 2 = 1]/@name",
    "(//Text)[1]/@name",
    # Axes, forward and reverse.
    "//TreeItem[@name='random']/following-sibling::TreeItem",
    "//TreeItem[@name='random']/following-sibling::TreeItem[1]/@name",
    "//TreeItem[@name='random']/preceding-sibling::TreeItem",
    "//TreeItem[@name='random']/preceding-sibling::TreeItem[1]/@name",
    "//Edit[@name='Message #general']/ancestor::Group[1]/@class",
    "//Edit[@name='Message #general']/ancestor::*",
    "//Edit[@name='Message #general']/ancestor-or-self::*[2]",
    "//Text[@name='Thanks! I will update the plan.']/../Button/@name",
    "//Button[@name='Bo Chen']/parent::*/parent::ListItem/@automation_id",
    "//List/self::List/@name",
    "//ListItem[1]/@*",
    "//Document[@automation_id='RootWebArea']/*/*",
    # Predicates.
    "//ListItem[.//Button[@name='Bo Chen']]/@automation_id",
    "//*[@class and not(@automation_id)][1]",
    "//ListItem[@automation_id and not(Document)]/@automation_id",
    "//TreeItem[contains(@name, 'a') and starts-with(@class, 'p-channel')]/@name",
    "//TreeItem[@name='general' or @name='random']",
    "//Group[Text]/@class",
    "//TreeItem | //TabItem",
    "//TabItem/@automation_id | //Tree/@name",
    # Functions and values.
    "count(//*)",
    "count(//ListItem[Document])",
    "string-length(//Tree/@name)",
    "substring(//Tree/@name, 1, 8)",
    "substring-before(//Window/@title, ' (')",
    "substring-after(//Window/@title, ' - ')",
    "normalize-space('  a  b  ')",
    "translate(//Tree/@name, 'aeiou', 'AEIOU')",
    "concat(count(//TreeItem), '/', count(//ListItem))",
    "name(//*[@automation_id='files'])",
    "//TreeItem/@name = 'random'",
    "//TreeItem/@name != 'random'",
    "count(//TreeItem) > count(//ListItem)",
    "not(//Slider)",
    "boolean(//Edit)",
    "sum(//TabItem[@automation_id='nope']/@name)",
    "floor(7 div 2) + ceiling(0.5) - round(2.5)",
    "3 mod 2 * -1",
    "string(//ListItem[2]/@automation_id)",
    "//ListItem[position() = 2 or position() = last()]/@automation_id",
]


def xml_tree(tree):
    """The first window as an lxml document, elements named by role."""
    window = tree["windows"][0]
    meta = window["window"]

    def element(recorded, parent):
        attrs = {}
        for key in ("name", "value", "class", "automation_id"):
            value = recorded.get(key)
            if value:
                attrs[key] = str(value)
        attrs["role"] = recorded["role"]
        attrs["password"] = "true" if recorded.get("password") else "false"
        node = etree.SubElement(parent, recorded["role"], attrs)
        for child in recorded.get("children", []):
            element(child, node)

    root = etree.Element(
        "Window",
        {
            "name": meta["title"],
            "role": "Window",
            "app": meta["app"],
            "title": meta["title"],
            "front": "true" if meta.get("front") else "false",
            "password": "false",
        },
    )
    for child in window.get("children", []):
        element(child, root)
    return etree.ElementTree(root)


def oracle(document, expression):
    """lxml's result, normalized: a list of ("element", role, name) / ("text", value), or a scalar."""
    result = document.xpath(expression)
    if isinstance(result, list):
        out = []
        for item in result:
            if isinstance(item, etree._Element):
                out.append(("element", item.tag, item.get("name", "")))
            else:
                out.append(("text", str(item)))
        return out
    return result


ELEMENT = re.compile(r'^(\w+)(?: "(.*?)")?(?: = ".*?")?(?: \.\S+)?(?: #\S+)?$')
WINDOW = re.compile(r'^Window "(.*)" \(.*\)$')


def jevons(binary, tree_file, expression):
    """jevons' result, normalized like the oracle's."""
    run = subprocess.run(
        [binary, "--tree", tree_file, "--xpath", expression],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        return ("error", run.stderr.strip())
    lines = [line for line in run.stdout.splitlines()]
    if "matches in" not in run.stderr:
        return lines[0] if lines else ""
    out = []
    for line in lines:
        if line.startswith('"'):
            out.append(("text", json.loads(line)))
        elif (window := WINDOW.match(line)) is not None:
            out.append(("element", "Window", window.group(1)))
        else:
            match = ELEMENT.match(line)
            if match is None:
                out.append(("unparsed", line))
            else:
                out.append(("element", match.group(1), match.group(2) or ""))
    return out


def same(expected, actual):
    if isinstance(expected, bool):
        return actual == ("true" if expected else "false")
    if isinstance(expected, float):
        try:
            number = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}.get(
                actual, None
            )
            number = float(actual) if number is None else number
        except (TypeError, ValueError):
            return False
        return (math.isnan(expected) and math.isnan(number)) or expected == number
    if isinstance(expected, str):
        return actual == expected
    return expected == actual


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    target = os.environ.get("CARGO_TARGET_DIR", os.path.join(root, "target"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bin", default=os.path.join(target, "debug", "jevons-desktop"))
    parser.add_argument(
        "--tree", default=os.path.join(root, "examples", "desktop", "trees", "slack.json")
    )
    args = parser.parse_args()
    with open(args.tree, encoding="utf-8") as f:
        document = xml_tree(json.load(f))
    failures = 0
    for expression in CORPUS:
        expected = oracle(document, expression)
        actual = jevons(args.bin, args.tree, expression)
        if same(expected, actual):
            print(f"ok    {expression}")
        else:
            failures += 1
            print(f"DIFF  {expression}\n      lxml:   {expected!r}\n      jevons: {actual!r}")
    print(f"\n{len(CORPUS) - failures} of {len(CORPUS)} agree")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
