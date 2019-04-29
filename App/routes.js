
Router.configure({
    layoutTemplate: 'mainLayout'
});


Router.route('/', {
    'name': 'smart_scan',
    template: 'main',
    onBeforeAction: function () { this.next(); },
    waitOn: function () {},
    data: function () {}
});
