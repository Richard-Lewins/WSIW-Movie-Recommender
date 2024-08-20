$(document).ready(function() {
    $('#search').on('keyup', function() {
        var query = $(this).val();
        
        // Update the URL without reloading the page
        var newUrl = window.location.protocol + "//" + window.location.host + window.location.pathname + '?q=' + query;
        history.pushState({path: newUrl}, '', newUrl);
        
        $.getJSON('/search', { q: query }, function(data) {
            var suggestions = $('#suggestions');
            suggestions.empty();
            
            if (data.length === 0) {
                suggestions.append('<div>No movie suggestions found. Please try a different search.</div>');
            } else {
                data.forEach(function(movie) {
                    suggestions.append('<div><a href="/recommendations/' + movie.movieId + '">' + movie.title + '</a></div>');
                });
            }
        });
    });

    // If there is a query in the URL when the page loads, trigger the search
    var urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('q')) {
        var query = urlParams.get('q');
        $('#search').val(query).trigger('keyup');
    }
});