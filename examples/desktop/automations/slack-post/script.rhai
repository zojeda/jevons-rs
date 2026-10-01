// Opens the conversation in the sidebar, types the message into the composer and sends it.

step(`opening ${args.channel}`);
let channel = find("//TreeItem[.//Group[has-class(@class, 'p-channel_sidebar__channel')]//Text[@name = $channel]]");
channel.invoke();

step("writing the message");
// The composer of the conversation just opened.
let composer = wait_for("//Edit[has-class(@class, 'ql-editor')]", 3000);
composer.type_text(args.message);
press("enter");

#{ posted: true, channel: channel.name }
