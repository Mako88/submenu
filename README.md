# submenu.js
=======

While building a website recently I realized I needed to be able
to add a submenu to the custom menu. To do this I hid a div and
then used Jquery to show and hide it on mouseover of the parent
element. So I decided to make it a plugin so others could use it.

## Usage

To use submenu.js, first link to Jquery, then to submenu.js, then
do:

```
(function() {
    $("#parent").submenu({
        sub: "#submenu",
        link: "#parentlink"
    });
});
```

The 'sub' option is to change the id or class name of the submenu div. The 'link' option is the (optional) link on the main menu (this should be an a tag).

If you need any help, let me know!
