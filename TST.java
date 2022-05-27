public class TST {
    private class Node{
        private Node left, mid, right;
        private Object val;
        private char c;
    }
    private Node root;
    public TST(){
        root = new Node();
    }
    public Object put(String key, Object val){
        root = put(root, key, 0, val);
        return null;
    }
    public Node put(Node node, String key, int d, Object val){
        char c = key.charAt(d);
        if (node == null){
            node = new Node();
            node.c = c;
        }
        if (c < node.c){
           node.left = put(node.left, key, d, val);
        }else if (c > node.c){
            node.right = put(node.right, key, d, val);
        }else if (d < key.length()-1){
            node.mid = put(node.mid, key, d+1, val);
        }else{
            node.val = val;
        }
        return node;
    }
    public Object get(String key){
        Node temp = get(root, key, 0);
        return temp.val;
    }
    public Node get(Node node, String key, int d){
        if (node == null) return null;
        char c = key.charAt(d);
        if (c < node.c){
            return get(node.left, key, d);
        }else if (c > node.c){
            return get(node.right, key, d);
        }else if (d < key.length()-1){
            return get(node.mid, key, d+1);
        }
        return node;
    }

}
