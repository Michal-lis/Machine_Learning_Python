
get_nth any [] = error "get nth"
get_nth 1 (front : any) = front
get_nth n (front : rest) = get_nth (n - 1) rest
