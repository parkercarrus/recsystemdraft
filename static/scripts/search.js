const search = document.querySelector('#name');
const data = document.querySelectorAll('.customer');


search.addEventListener("input", (e) => {
    const query = e.target.value.toLowerCase();
    if (query) {
        data.forEach((customer) => { 
            if (customer.innerHTML.toLowerCase().includes(query)) {customer.style.display='block'} else (customer.style.display='none')
            });  // each time the search bar receives an input, 
    // filter through all the data and return all names that contains whats in the query 
        }
    else {
        data.forEach((customer) => customer.style.display ='none')
    }
});
