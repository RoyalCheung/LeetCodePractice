import java.io.ObjectStreamException;
import java.util.ArrayDeque;
import java.util.Queue;

public class Trie {
    private class Node{
        private Object val;
        private Node[] next;
        Node(int R){
            next = new Node[R];
        }
        Node(){
            next = new Node[256];
        }
    }
    private Node root;
    private int R;
    public Trie(int R){
        root = new Node(R);
        this.R = R;
    }
    public Trie(){
        root = new Node();
        this.R = 256;
    }
    public void insert(String key, Object val){
        root = insert(root, key, 0, val);
    }
    public Node insert(Node node, String key, int d, Object val){
        if (node == null) node = new Node(R);
        if (d == key.length()) {
            node.val = val;
            return node;
        }
        char c = key.charAt(d);
        node.next[c] = insert(node.next[c], key, d+1, val);
        return node;
    }
    public Object search(String key){
        Node node = search(root, key, 0);

        return node.val;
    }
    public Node search(Node node, String key, int d){
        if (node == null) return null;
        if (d == key.length()) return node;
        char c = key.charAt(d);
        return search(node.next[c], key, d+1);
    }

    public void collect(Node x, String pre, Queue<String> q){
        if (x == null) return;
        if (x.val != null) q.add(pre);
        for(char c = 0; c < R; c++){
            collect(x.next[c], pre+c, q);
        }
    }
    public Iterable<String> keysWithPrefix(String pre){
        Queue<String> queue = new ArrayDeque<>();
        collect(search(root, pre, 0), pre, queue);
        return queue;
    }
    public Iterable<String> keys(){
        return keysWithPrefix("");
    }

    public Iterable<String> keysThatMatch(String pat) {
        Queue<String> queue = new ArrayDeque<>();
        collect(root, "", pat, queue);
        return queue;
    }
    public void collect(Node node, String key, String pat, Queue<String> queue){
        int d = key.length();
        if (node == null) return;
        if (d == pat.length() && node.val != null) queue.add(key);
        if (d == pat.length()) return;
        char next = pat.charAt(d);
        for (char c = 0; c < R; c++){
            if (next == '.' || next == c){
                collect(node.next[c], key + c, pat, queue);
            }
        }
    }
    public String longestPreOf(String s){
        int length = search(root, s, 0, 0);
        return s.substring(0, length);
    }
    public int search(Node node, String key, int d, int length){
        if (node == null) return length;
        if (node.val != null) length = d;
        if (d == key.length()) return length;
        char c = key.charAt(d);
        return search(node.next[c], key, d+1, length);
    }
    public void delete(String key){
        root = delete(root, key, 0);
    }
    public Node delete(Node node, String key, int d){
        if (node == null) return null;
        if (d == key.length()){
            node.val = null;
        }else {
            char c = key.charAt(d);
            node.next[c] = delete(node.next[c], key, d + 1);
        }
        if (node.val != null) return node;
        for (char c = 0; c < R; c++){
            if (node.next[c] != null) return node;
        }
        return null;
    }

    public static void main(String[] args) {
        Trie trie = new Trie();
        trie.insert("apple", 26);
        trie.insert("all", 27);
        Iterable<String> queue = trie.keys();
        System.out.println(queue);
        queue = trie.keysThatMatch("a.l");
        System.out.println(queue);
        char a = 98;
        String pre = "A" + a;
        System.out.println(pre);
        System.out.println(trie.longestPreOf("allstar"));
//        System.out.println(trie.search("apple"));

    }
}
