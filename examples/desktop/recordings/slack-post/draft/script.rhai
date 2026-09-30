// Post lunch is ready to the random channel
// Written by jevons from a recording; edit it freely (it then needs approving again).

step("choosing the item");
let e1 = find("//TreeItem[.//Text[@name = $item]]");
e1.invoke();

step("typing the text");
let e2 = wait_for("//Edit[has-class(@class, 'ql-editor')]", 5000);
e2.type_text(args.text);

step("pressing enter");
press("enter");

#{ done: true }
