$('.fupload').change(function(e) {

    for (var i = 0; i < e.originalEvent.srcElement.files.length; i++) {
        
        var file = e.originalEvent.srcElement.files[i];
        console.log(file)
        
        var img = document.createElement("img");
        var reader = new FileReader();
        reader.onloadend = function() {
            img.src = reader.result;
            img.width = 200
        }
        reader.readAsDataURL(file);
        $('.dimg').html(img);
    }

    let fn = e.originalEvent.srcElement.files[0].name
    console.log(fn)
    $.ajax({url: `/make_inference/${fn}`,

        success: function(result) {
            let res = result.res
            $('.pred').html(res)
        }

    })

});