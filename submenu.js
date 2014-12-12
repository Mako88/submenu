/*
 *  jQuery Boilerplate - v3.3.2
 *  A jump-start for jQuery plugins development.
 *  http://jqueryboilerplate.com
 *
 *  Made by Zeno Rocha
 *  Under MIT License
 */
// the semi-colon before function invocation is a safety net against concatenated
// scripts and/or other plugins which may not be closed properly.
;(function ( $, window, document, undefined ) {

	// undefined is used here as the undefined global variable in ECMAScript 3 is
	// mutable (ie. it can be changed by someone else). undefined isn't really being
	// passed in so we can ensure the value of it is truly undefined. In ES5, undefined
	// can no longer be modified.

	// window and document are passed through as local variable rather than global
	// as this (slightly) quickens the resolution process and can be more efficiently
	// minified (especially when both are regularly referenced in your plugin).

	// Create the defaults once
	var pluginName = "submenu",
		defaults = {
		sub: "#submenu",
        	link: "#parentlink"
	};

	// The actual plugin constructor
	function Plugin ( element, options ) {
		this.element = element;
		// jQuery has an extend method which merges the contents of two or
		// more objects, storing the result in the first object. The first object
		// is generally empty as we don't want to alter the default options for
		// future instances of the plugin
		this.settings = $.extend({}, defaults, options);
        	this.settings.sub = $(this.settings.sub);
        	this.settings.link = $(this.settings.link);
		this._defaults = defaults;
		this._name = pluginName;
		this.init();
	}

	Plugin.prototype = {
		init: function () {
			// Place initialization logic here
			// You already have access to the DOM element and
			// the options via the instance, e.g. this.element
			// and this.settings
			// you can add more functions like the one below and
			// call them like so: this.yourOtherFunction(this.element, this.settings).
                        var sub = this.settings.sub, link = this.settings.link, parentmouse = false, submouse = false, subactive = false;
                        if('ontouchstart' in window || navigator.msMaxTouchPoints) {
                            $(this.element).on("touchstart", function () {
                                if (subactive == false) {
                                    link.on("click", function (e) {
                                        e.preventDefault();
                                    });
                                    sub.css("z-index", "3");
                                    sub.fadeIn("normal");
                                    subactive = true;
                                }
                                else {
                                    window.location = link.attr('href');
                                }
                            });
                            $("html").on("touchstart", function (e) {
                                if($(e.target).closest(sub).length != 0) return true;
                                if($(e.target).closest(link).length != 0) return true;
                                if (subactive == true) {
                                    sub.fadeOut("normal", function () {
                                        sub.css("z-index", "-1");
                                    });
                                    subactive = false;
                                }
                            });
                        }
                        else {
                            $(this.element).on("mouseenter", function () {
                                sub.stop(true, true);
                                if (parentmouse === false) {
                                    sub.css("z-index", "3");
                                    sub.fadeIn("normal");
                                }
                                parentmouse = true;
                            });
                            $(this.element).on("mouseleave", function () {
                                setTimeout(function () {
                                    if (submouse === false) {
                                        sub.fadeOut("normal", function () {
                                            sub.css("z-index", "-1");
                                        });
                                    }
                                }, 100);
                                parentmouse = false;
                            });
                            sub.on("mouseenter", function () {
                                submouse = true;
                                sub.stop(true, true);
                                sub.css({"z-index": "3", "display": "block"});
                            });
                            sub.on("mouseleave", function () {
                                setTimeout(function () {
                                    if (parentmouse === false) {
                                        sub.fadeOut("normal", function () {
                                            sub.css("z-index", "-1");
                                        });
                                    }
                                }, 100);
                                submouse = false;
                            });
                        }
		},
	};

	// A really lightweight plugin wrapper around the constructor,
	// preventing against multiple instantiations
	$.fn[ pluginName ] = function ( options ) {
		this.each(function() {
			if ( !$.data( this, "plugin_" + pluginName ) ) {
				$.data( this, "plugin_" + pluginName, new Plugin( this, options ) );
			}
		});

		// chain jQuery functions
		return this;
	};

})( jQuery, window, document );
